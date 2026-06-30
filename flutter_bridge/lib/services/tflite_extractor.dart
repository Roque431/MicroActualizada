/// Extractor clínico on-device para EpiDiagnostix-Mayab.
///
/// Pipeline completo (offline, sin servidor):
///   texto
///     → tokenize()           [clinical_tokenizer.dart — paridad 35/35 vs Python]
///     → TFLite BiLSTM NER    [models/ner_tflite_v2.tflite — built-in ops]
///     → applyRules()         [bio_post_processor.dart — paridad 25/25 vs Python]
///     → extractFields()      [bio_post_processor.dart — paridad 25/25 vs Python]
///     → Map<String, dynamic> [mismo formato que ConsultaResult del microservicio]
///
/// Uso en Flutter:
///
///   // Inicializar una vez (en initState / lazy singleton)
///   final extractor = await NerExtractor.fromAssets();
///
///   // Inferir en cada texto
///   final campos = extractor.infer(texto);
///   // {'edad': 45, 'sexo': 'M', 'presion_sistolica': 125, ...}
///
/// Requisitos en el app Flutter:
///   pubspec.yaml:
///     dependencies:
///       tflite_flutter: ^0.10.4
///     flutter:
///       assets:
///         - models/ner_tflite_v2.tflite
///         - models/tflite_vocab_v2.json
///         - models/tflite_labels_v2.json
///
///   android/app/build.gradle:
///     android { aaptOptions { noCompress 'tflite' } }

library tflite_extractor;

import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';

import 'clinical_tokenizer.dart';
import 'bio_post_processor.dart';

const int _maxLen = 40;

/// Servicio de inferencia NER on-device.
///
/// Carga el modelo TFLite y el vocabulario una sola vez;
/// [infer] puede llamarse tantas veces como sea necesario.
class NerExtractor {
  final Interpreter _interpreter;
  final Map<String, int> _vocab;
  final Map<int, String> _id2Label;
  final int _nLabels;
  final int _unkId;

  NerExtractor._({
    required Interpreter interpreter,
    required Map<String, int> vocab,
    required Map<int, String> id2Label,
  })  : _interpreter = interpreter,
        _vocab = vocab,
        _id2Label = id2Label,
        _nLabels = id2Label.length,
        _unkId = vocab['<UNK>'] ?? 1;

  // ── Constructores ──────────────────────────────────────────────────────────

  /// Carga el modelo y vocabulario desde los assets del app Flutter.
  /// Llamar una sola vez durante el startup (e.g. en initState o un Provider).
  static Future<NerExtractor> fromAssets({
    String modelAsset = 'models/ner_tflite_v2.tflite',
    String vocabAsset = 'models/tflite_vocab_v2.json',
    String labelsAsset = 'models/tflite_labels_v2.json',
  }) async {
    final modelBytes = await _loadAssetBytes(modelAsset);
    final vocabJson  = await rootBundle.loadString(vocabAsset);
    final labelsJson = await rootBundle.loadString(labelsAsset);
    return fromBytes(
      modelBytes: modelBytes,
      vocabJson: vocabJson,
      labelsJson: labelsJson,
    );
  }

  /// Constructor alternativo para tests: carga desde bytes y strings en memoria.
  /// Permite probar sin assets de Flutter (e.g., cargando desde File en Dart tests).
  static NerExtractor fromBytes({
    required Uint8List modelBytes,
    required String vocabJson,
    required String labelsJson,
  }) {
    final interpreter = Interpreter.fromBuffer(modelBytes);

    // Verificar shapes de I/O
    final inpShape = interpreter.getInputTensors()[0].shape;
    final outShape = interpreter.getOutputTensors()[0].shape;
    assert(inpShape[1] == _maxLen,
        'Input shape esperado [1,$_maxLen], obtenido $inpShape');
    assert(outShape[1] == _maxLen,
        'Output shape esperado [1,$_maxLen,N], obtenido $outShape');

    // Vocabulario: token → id
    final vocabRaw = jsonDecode(vocabJson) as Map<String, dynamic>;
    final vocab = vocabRaw.map((k, v) => MapEntry(k, v as int));

    // Etiquetas: id → label
    final labelsRaw = jsonDecode(labelsJson) as Map<String, dynamic>;
    final id2Label  = labelsRaw.map((k, v) => MapEntry(v as int, k));

    return NerExtractor._(
      interpreter: interpreter,
      vocab: vocab,
      id2Label: id2Label,
    );
  }

  // ── Pipeline principal ─────────────────────────────────────────────────────

  /// Extrae campos clínicos estructurados del [text].
  ///
  /// Retorna un Map con los campos presentes y con valor válido:
  /// - 'edad'                   → int
  /// - 'sexo'                   → 'M' | 'F'
  /// - 'peso_kg'                → double
  /// - 'talla_cm'               → double
  /// - 'presion_sistolica'      → int
  /// - 'presion_diastolica'     → int
  /// - 'glucosa_mg_dl'          → int
  /// - 'temperatura_c'          → double
  /// - 'frecuencia_cardiaca_bpm'→ int
  /// - 'duracion_sintomas_dias' → int
  /// - 'categoria_sintoma'      → String
  ///
  /// Los campos no detectados simplemente no aparecen en el mapa.
  Map<String, dynamic> infer(String text) {
    // 1. Tokenizar (mismo regex que Python: \d+[.,]\d+|\d+|[a-z]+)
    final tokens = tokenize(text).take(_maxLen).toList();
    if (tokens.isEmpty) return {};

    // 2. Codificar con padding hasta maxLen
    final ids = tokensToIds(tokens, _vocab);
    final padded = List<int>.from(ids)
      ..addAll(List.filled(_maxLen - ids.length, padId));

    // 3. Inferencia TFLite
    //    Input  : [1, 40] int32
    //    Output : [1, 40, nLabels] float32
    final input  = [padded];
    final output = [
      List.generate(_maxLen, (_) => List<double>.filled(_nLabels, 0.0)),
    ];
    _interpreter.run(input, output);

    // 4. Argmax → etiquetas crudas (solo para la longitud real del texto)
    final labelsRaw = _argmaxLabels(output[0], tokens.length);

    // 5. Post-procesamiento: R1 + R2 (mutación de etiquetas)
    final labelsPp = applyRules(tokens, labelsRaw);

    // 6. Extracción de campos: R3 + extractor completo
    return extractFields(tokens, labelsPp);
  }

  /// Libera los recursos del intérprete TFLite.
  void dispose() => _interpreter.close();

  // ── Helpers privados ───────────────────────────────────────────────────────

  List<String> _argmaxLabels(List<List<double>> logits, int nTokens) {
    return List.generate(nTokens, (i) {
      final row = logits[i];
      int best = 0;
      for (int j = 1; j < row.length; j++) {
        if (row[j] > row[best]) best = j;
      }
      return _id2Label[best] ?? 'O';
    });
  }

  static Future<Uint8List> _loadAssetBytes(String asset) async {
    final data = await rootBundle.load(asset);
    return data.buffer.asUint8List();
  }
}

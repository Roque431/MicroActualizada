/// Prueba de paridad: tokenizador Dart vs. tokenizador Python.
///
/// Carga models/tokenizer_golden_set.json (generado por generar_golden_set.py)
/// y verifica frase por frase que Dart produce EXACTAMENTE los mismos tokens
/// e IDs que Python.
///
/// Ejecutar:
///   dart test test/tokenizer_parity_test.dart
///   flutter test test/tokenizer_parity_test.dart
///
/// PASA solo si todas las frases coinciden al 100%.

import 'dart:convert';
import 'dart:io';
import 'package:test/test.dart';
import '../lib/services/clinical_tokenizer.dart';

void main() {
  late List<Map<String, dynamic>> goldenFrases;
  late Map<String, int> vocab;

  setUpAll(() {
    // Cargar golden set generado por Python
    final goldenFile = File('../models/tokenizer_golden_set.json');
    final goldenJson = jsonDecode(goldenFile.readAsStringSync()) as Map;
    goldenFrases = List<Map<String, dynamic>>.from(goldenJson['frases'] as List);

    // Cargar vocabulario
    final vocabFile = File('../models/tflite_vocab_v2.json');
    final vocabJson = jsonDecode(vocabFile.readAsStringSync()) as Map;
    vocab = vocabJson.map((k, v) => MapEntry(k as String, v as int));
  });

  group('Paridad tokenizador Python → Dart', () {
    test('Golden set tiene 35 frases', () {
      expect(goldenFrases.length, equals(35));
    });

    int coincidencias = 0;
    int total = 0;
    final fallos = <Map<String, dynamic>>[];

    // Test individual por frase
    for (var i = 0; i < 35; i++) {
      test('Frase ${i + 1}: ${_preview(i)}', () {
        final frase = goldenFrases[i];
        final original = frase['original'] as String;
        final tokensEsperados = List<String>.from(frase['tokens'] as List);
        final idsEsperados = List<int>.from(frase['ids'] as List);

        // Tokenizar con Dart
        final tokensDart = tokenize(original);
        final idsDart = tokensToIds(tokensDart, vocab);

        total++;

        // Comparar tokens
        if (tokensDart.length != tokensEsperados.length) {
          fallos.add({
            'frase': i + 1,
            'original': original,
            'error': 'Longitudes distintas: Dart=${tokensDart.length} Python=${tokensEsperados.length}',
            'dart': tokensDart,
            'python': tokensEsperados,
          });
        } else {
          bool tokensIguales = true;
          int? primerFallo;
          for (var j = 0; j < tokensDart.length; j++) {
            if (tokensDart[j] != tokensEsperados[j]) {
              tokensIguales = false;
              primerFallo = j;
              break;
            }
          }
          if (!tokensIguales) {
            fallos.add({
              'frase': i + 1,
              'original': original,
              'error': 'Token[$primerFallo] difiere: '
                  'Dart="${tokensDart[primerFallo!]}" Python="${tokensEsperados[primerFallo]}"',
              'dart': tokensDart,
              'python': tokensEsperados,
            });
          } else {
            coincidencias++;
          }
        }

        // Assertions
        expect(
          tokensDart,
          equals(tokensEsperados),
          reason: 'Tokens difieren para: "$original"',
        );
        expect(
          idsDart,
          equals(idsEsperados),
          reason: 'IDs difieren para: "$original"',
        );
      });
    }

    test('RESUMEN — todas las frases coinciden', () {
      if (fallos.isNotEmpty) {
        final msg = StringBuffer('${fallos.length} frases no coinciden:\n');
        for (final f in fallos) {
          msg.writeln('  Frase ${f['frase']}: ${f['error']}');
          msg.writeln('    Original: ${f['original']}');
          msg.writeln('    Dart:   ${f['dart']}');
          msg.writeln('    Python: ${f['python']}');
        }
        fail(msg.toString());
      }
      print('\nPARIDAD: $coincidencias/$total frases — 100% coincidencia');
    });
  });
}

String _preview(int i) {
  // Placeholder — se reemplaza con el original al ejecutar
  return 'frase $i (ver golden set)';
}

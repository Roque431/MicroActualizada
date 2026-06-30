/// Post-procesador BIO para EpiDiagnostix-Mayab.
///
/// Replica EXACTAMENTE la lógica Python validada en postprocess_rules_v2.py:
///
///   aplicar_reglas_1_2(tokens, labels)  →  applyRules()
///   extraer(tokens, labels)             →  extractFields()
///
/// Orden de aplicación obligatorio:
///   1. applyRules()     → R2 primero, luego R1 (muta etiquetas)
///   2. extractFields()  → R3 integrada en el extractor
///
/// Resultado final de extraer campos clínicos (sin null):
///   edad, sexo, peso_kg, talla_cm, presion_sistolica, presion_diastolica,
///   glucosa_mg_dl, temperatura_c, frecuencia_cardiaca_bpm,
///   duracion_sintomas_dias, categoria_sintoma

library bio_post_processor;

import 'dart:math' show max, min;

// ── Constantes (idénticas a Python) ─────────────────────────────────────────

const Set<String> _sexoTokens = {
  'mujer', 'hombre', 'varon', 'masculino', 'femenino', 'femenina',
  'senora', 'senor', 'nina', 'nino', 'dama', 'caballero', 'muchacha', 'muchacho',
};

const Set<String> _durContext = {
  'dias', 'dia', 'evolucion', 'cuadro', 'lleva', 'desde', 'hace',
};

const Map<String, String> _sexoMap = {
  'masculino': 'M', 'hombre': 'M', 'varon': 'M',
  'nino': 'M', 'senor': 'M', 'caballero': 'M',
  'mujer': 'F', 'femenino': 'F', 'femenina': 'F',
  'nina': 'F', 'senora': 'F', 'dama': 'F',
};

const Map<String, String> _catMap = {
  'gastrointestinal': 'Gastrointestinal',
  'respiratorio': 'Respiratorio',
  'hipertension': 'Hipertension',
  'diabetes': 'Diabetes',
  'dengue': 'Infeccioso/Vectorial',
  'vacunacion': 'Vacunacion',
  'nutricion': 'Nutricion',
};

// Coincide con Python str.isdigit() para tokens ASCII del tokenizador.
final RegExp _digitsOnly = RegExp(r'^\d+$');

// ── R1 + R2: mutación de etiquetas ───────────────────────────────────────────

/// Aplica R2 (número atrapado en span SEXO → B-EDAD) y R1 (segundo B-EDAD
/// con contexto de duración → B-DURACION) sobre una copia de [labelsIn].
///
/// Equivalente Python: aplicar_reglas_1_2(tokens, labels)
/// Orden obligatorio: R2 primero, R1 segundo.
List<String> applyRules(List<String> tokens, List<String> labelsIn) {
  final labels = List<String>.from(labelsIn);

  // R2: dígito etiquetado como B-SEXO cuando antes hay un token de género
  //     y después hay "anos/an/a" como I-SEXO → corregir a B-EDAD + O.
  for (int i = 0; i < labels.length; i++) {
    if (labels[i] == 'B-SEXO' && _digitsOnly.hasMatch(tokens[i])) {
      bool hayGenero = false;
      for (int j = max(0, i - 5); j < i; j++) {
        if ((labels[j] == 'B-SEXO' || labels[j] == 'I-SEXO') &&
            _sexoTokens.contains(tokens[j])) {
          hayGenero = true;
          break;
        }
      }
      final sigAnos = i + 1 < tokens.length &&
          labels[i + 1] == 'I-SEXO' &&
          (tokens[i + 1] == 'anos' || tokens[i + 1] == 'an' || tokens[i + 1] == 'a');
      if (hayGenero && sigAnos) {
        labels[i] = 'B-EDAD';
        labels[i + 1] = 'O';
      }
    }
  }

  // R1: cuando hay más de un B-EDAD, los duplicados con tokens de duración
  //     en ventana ±4 se relabelen como B-DURACION.
  final edadIdx = <int>[
    for (int i = 0; i < labels.length; i++)
      if (labels[i] == 'B-EDAD') i,
  ];
  if (edadIdx.length > 1) {
    for (int k = 1; k < edadIdx.length; k++) {
      final idx = edadIdx[k];
      final ini = max(0, idx - 4);
      final fin = min(tokens.length, idx + 5);
      final ctx = tokens.sublist(ini, fin).toSet()..remove(tokens[idx]);
      if (ctx.intersection(_durContext).isNotEmpty) {
        labels[idx] = 'B-DURACION';
      }
    }
  }
  return labels;
}

// ── Conversores numéricos ────────────────────────────────────────────────────

/// Equivalente a Python: int(float(str(v).replace(",",".")))
int? _toInt(String tok) {
  final d = double.tryParse(tok.replaceAll(',', '.'));
  return d?.truncate();
}

/// Equivalente a Python: round(float(str(v).replace(",",".")), 1)
double? _toFloat(String tok) {
  final d = double.tryParse(tok.replaceAll(',', '.'));
  if (d == null) return null;
  return double.parse(d.toStringAsFixed(1));
}

// ── Extractor con R3 integrada ───────────────────────────────────────────────

/// Extrae campos clínicos estructurados de [tokens]/[labels] post-procesados.
/// R3 se aplica internamente: itera todos los tokens con "SEXO" en la etiqueta
/// y toma el primero con valor válido en sexoMap.
///
/// Solo se incluyen campos con valor no-nulo en el mapa devuelto.
/// Equivalente Python: extraer(tokens, labels)
Map<String, dynamic> extractFields(List<String> tokens, List<String> labels) {
  final res = <String, dynamic>{};

  // R3: SEXO — primer token (B-SEXO o I-SEXO) con valor válido en sexoMap.
  for (int i = 0; i < tokens.length; i++) {
    if (labels[i].contains('SEXO')) {
      final v = _sexoMap[tokens[i]];
      if (v != null) {
        res['sexo'] = v;
        break;
      }
    }
  }

  // Campos restantes: solo tokens con etiqueta B-.
  for (int i = 0; i < tokens.length; i++) {
    final lbl = labels[i];
    if (lbl == 'O' || !lbl.startsWith('B-')) continue;
    final base = lbl.substring(2);
    final tok = tokens[i];

    switch (base) {
      case 'EDAD':
        if (!res.containsKey('edad')) {
          final v = _toInt(tok);
          if (v != null && v > 0 && v <= 120) res['edad'] = v;
        }
      case 'PESO_KG':
        if (!res.containsKey('peso_kg')) {
          final v = _toFloat(tok);
          if (v != null) res['peso_kg'] = v;
        }
      case 'TALLA_CM':
        if (!res.containsKey('talla_cm')) {
          final v = _toFloat(tok);
          // v < 3 significa que está en metros (ej. 1.53) → convertir a cm
          if (v != null) {
            res['talla_cm'] = v < 3
                ? double.parse((v * 100).toStringAsFixed(1))
                : v;
          }
        }
      case 'PRESION_SIS':
        if (!res.containsKey('presion_sistolica')) {
          final v = _toInt(tok);
          if (v != null) res['presion_sistolica'] = v;
        }
      case 'PRESION_DIA':
        if (!res.containsKey('presion_diastolica')) {
          final v = _toInt(tok);
          if (v != null) res['presion_diastolica'] = v;
        }
      case 'GLUCOSA':
        if (!res.containsKey('glucosa_mg_dl')) {
          final v = _toInt(tok);
          if (v != null) res['glucosa_mg_dl'] = v;
        }
      case 'TEMPERATURA':
        if (!res.containsKey('temperatura_c')) {
          final v = _toFloat(tok);
          if (v != null) res['temperatura_c'] = v;
        }
      case 'FREC_CARD':
        if (!res.containsKey('frecuencia_cardiaca_bpm')) {
          final v = _toInt(tok);
          if (v != null) res['frecuencia_cardiaca_bpm'] = v;
        }
      case 'DURACION':
        if (!res.containsKey('duracion_sintomas_dias')) {
          final v = _toInt(tok);
          if (v != null) res['duracion_sintomas_dias'] = v;
        }
      case 'CATEGORIA':
        if (!res.containsKey('categoria_sintoma')) {
          final v = _catMap[tok];
          if (v != null) res['categoria_sintoma'] = v;
        }
    }
  }
  return res;
}

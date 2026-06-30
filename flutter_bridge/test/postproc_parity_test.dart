/// Prueba de paridad: post-procesador Dart vs. post-procesador Python.
///
/// Carga models/postproc_golden_set.json (generado por generar_golden_set_postproc.py)
/// y verifica caso por caso que Dart produce:
///   (a) EXACTAMENTE las mismas etiquetas post-procesadas (labels_pp) que Python
///       al llamar a applyRules(tokens, labels_raw).
///   (b) EXACTAMENTE los mismos campos extraídos (fields) que Python
///       al llamar a extractFields(tokens, labels_pp).
///
/// El test NO pasa hasta que las 25 casos coincidan al 100% en ambas métricas.
///
/// Ejecutar:
///   cd flutter_bridge
///   dart test test/postproc_parity_test.dart

import 'dart:convert';
import 'dart:io';
import 'package:test/test.dart';
import '../lib/services/bio_post_processor.dart';

void main() {
  late List<Map<String, dynamic>> goldenCases;

  setUpAll(() {
    final f = File('../models/postproc_golden_set.json');
    final raw = jsonDecode(f.readAsStringSync()) as Map<String, dynamic>;
    goldenCases = List<Map<String, dynamic>>.from(raw['cases'] as List);
  });

  // ── 1. Metadatos ──────────────────────────────────────────────────────────

  test('Golden set tiene 25 casos (5 SET2 + 20 SET3)', () {
    expect(goldenCases.length, equals(25));
  });

  // ── 2. Paridad de etiquetas (R1+R2) ───────────────────────────────────────

  group('applyRules — paridad etiquetas Python → Dart', () {
    // Contadores acumulados (visibles en el resumen final)
    int okLabels = 0;
    int totalLabels = 0;
    final labelFails = <String>[];

    for (var i = 0; i < 25; i++) {
      test('labels_pp [$i]: ${_caseId(i)}', () {
        final c = goldenCases[i];
        final tokens    = _strings(c['tokens']);
        final labelsRaw = _strings(c['labels_raw']);
        final labelsPy  = _strings(c['labels_pp']);

        final labelsDart = applyRules(tokens, labelsRaw);
        totalLabels++;

        if (!_listEq(labelsDart, labelsPy)) {
          final diff = _labelDiff(tokens, labelsRaw, labelsPy, labelsDart);
          labelFails.add('[${c['id']}] $diff');
        } else {
          okLabels++;
        }

        expect(
          labelsDart,
          equals(labelsPy),
          reason: 'Etiquetas difieren para caso "${c['id']}"\n'
              'tokens:  $tokens\n'
              'raw:     $labelsRaw\n'
              'python:  $labelsPy\n'
              'dart:    $labelsDart',
        );
      });
    }

    test('RESUMEN labels — 25/25 casos coinciden', () {
      if (labelFails.isNotEmpty) {
        fail('${labelFails.length} caso(s) con etiquetas distintas:\n'
            '${labelFails.join('\n')}');
      }
      // ignore: avoid_print
      print('\nPARIDAD LABELS: $okLabels/$totalLabels casos — 100%');
    });
  });

  // ── 3. Paridad de campos extraídos (R3 + extractor) ───────────────────────

  group('extractFields — paridad campos Python → Dart', () {
    int okFields = 0;
    int totalFields = 0;
    final fieldFails = <String>[];

    for (var i = 0; i < 25; i++) {
      test('fields [$i]: ${_caseId(i)}', () {
        final c = goldenCases[i];
        final tokens   = _strings(c['tokens']);
        final labelsPp = _strings(c['labels_pp']);  // usa labels ya post-procesadas
        final fieldsPy = Map<String, dynamic>.from(c['fields'] as Map);

        final fieldsDart = extractFields(tokens, labelsPp);
        totalFields++;

        final diffs = _fieldDiff(fieldsPy, fieldsDart);
        if (diffs.isEmpty) {
          okFields++;
        } else {
          fieldFails.add('[${c['id']}] ${diffs.join(', ')}');
        }

        // Verificar que cada campo esperado está presente y es correcto
        for (final entry in fieldsPy.entries) {
          final campo = entry.key;
          final espPy  = entry.value;
          final obtDart = fieldsDart[campo];

          expect(
            obtDart,
            isNotNull,
            reason: 'Campo "$campo" ausente en Dart para caso "${c['id']}"',
          );

          if (espPy is num && obtDart is num) {
            // Comparación numérica exacta (generator y Dart usan la misma lógica)
            expect(
              obtDart.toDouble(),
              closeTo(espPy.toDouble(), 0.05),
              reason: 'Campo "$campo": python=$espPy dart=$obtDart '
                  'para caso "${c['id']}"',
            );
          } else {
            expect(
              obtDart.toString(),
              equals(espPy.toString()),
              reason: 'Campo "$campo": python="$espPy" dart="$obtDart" '
                  'para caso "${c['id']}"',
            );
          }
        }

        // No debe haber campos extra que Python no produjo
        for (final dartKey in fieldsDart.keys) {
          expect(
            fieldsPy.containsKey(dartKey),
            isTrue,
            reason: 'Campo extra en Dart no presente en Python: '
                '"$dartKey" para caso "${c['id']}"',
          );
        }
      });
    }

    test('RESUMEN fields — 25/25 casos coinciden', () {
      if (fieldFails.isNotEmpty) {
        fail('${fieldFails.length} caso(s) con campos distintos:\n'
            '${fieldFails.join('\n')}');
      }
      // ignore: avoid_print
      print('\nPARIDAD FIELDS: $okFields/$totalFields casos — 100%');
    });
  });
}

// ── Helpers ──────────────────────────────────────────────────────────────────

String _caseId(int i) {
  // Solo disponible dentro del test después de setUpAll; aquí mostramos índice
  return 'caso $i';
}

List<String> _strings(dynamic raw) =>
    List<String>.from((raw as List).map((e) => e.toString()));

bool _listEq(List<String> a, List<String> b) {
  if (a.length != b.length) return false;
  for (int i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

String _labelDiff(
  List<String> tokens,
  List<String> raw,
  List<String> python,
  List<String> dart,
) {
  final diffs = <String>[];
  for (int i = 0; i < python.length; i++) {
    if (i < dart.length && python[i] != dart[i]) {
      diffs.add('[${i}]"${tokens[i]}": py=${python[i]} dart=${dart[i]}');
    }
  }
  return diffs.join('; ');
}

List<String> _fieldDiff(
  Map<String, dynamic> python,
  Map<String, dynamic> dart,
) {
  final diffs = <String>[];
  for (final k in python.keys) {
    if (!dart.containsKey(k)) {
      diffs.add('$k: ausente en Dart (python=${python[k]})');
    } else {
      final py = python[k];
      final d  = dart[k];
      if (py is num && d is num) {
        if ((py.toDouble() - d.toDouble()).abs() > 0.05) {
          diffs.add('$k: py=$py dart=$d');
        }
      } else if (py.toString() != d.toString()) {
        diffs.add('$k: py="$py" dart="$d"');
      }
    }
  }
  for (final k in dart.keys) {
    if (!python.containsKey(k)) {
      diffs.add('$k: extra en Dart (dart=${dart[k]})');
    }
  }
  return diffs;
}

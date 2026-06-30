/// Tokenizador clínico para EpiDiagnostix-Mayab.
///
/// Replica EXACTAMENTE la función Python del dataset de entrenamiento:
///
///   _TOKEN_RE = re.compile(r'\d+[.,]\d+|\d+|[a-z]+')
///   def tokenizar(texto):
///       return [m.group() for m in _TOKEN_RE.finditer(texto.lower())]
///
/// Reglas de tokenización (idénticas a Python):
///   1. El texto se convierte a minúsculas completas (.toLowerCase())
///   2. El regex captura, en orden de prioridad:
///      a) Números decimales: uno o más dígitos + '.' o ',' + uno o más dígitos
///         Ejemplos: "37.5" → ["37.5"],  "37,5" → ["37,5"]
///      b) Números enteros:  uno o más dígitos
///         Ejemplos: "120" → ["120"]
///      c) Palabras ASCII:   uno o más caracteres [a-z] (sólo ASCII a-z)
///         Ejemplos: "temperatura" → ["temperatura"]
///   3. TODO lo demás se descarta: /, -, °, :, ;, !, ?, espacios, acentos...
///      Los caracteres acentuados (á é í ó ú ñ ü) NO son [a-z] → dividen palabras
///      Ejemplos: "presión" → ["presi","n"],  "años" → ["a","os"]
///
/// El vocabulario (<PAD>=0, <UNK>=1, resto ≥2) se carga del JSON generado
/// por el pipeline de entrenamiento de Python.

library clinical_tokenizer;

/// Patrón idéntico al de Python, compilado una sola vez.
/// En Dart, [a-z] también es ASCII-only por defecto.
final RegExp _tokenRe = RegExp(r'\d+[.,]\d+|\d+|[a-z]+');

const int padId = 0;
const int unkId = 1;

/// Tokeniza [text] y devuelve la lista de tokens.
/// Equivalente al `tokenizar(texto)` de Python.
List<String> tokenize(String text) {
  final textLower = text.toLowerCase();
  return _tokenRe
      .allMatches(textLower)
      .map((m) => m.group(0)!)
      .toList();
}

/// Convierte una lista de tokens en IDs del vocabulario.
/// Tokens desconocidos reciben [unkId] (1).
List<int> tokensToIds(List<String> tokens, Map<String, int> vocab) {
  return tokens.map((t) => vocab[t] ?? unkId).toList();
}

/// Pipeline completa: texto → IDs con padding hasta [maxLen].
/// Equivalente al `cargar_y_padear()` del entrenamiento.
List<int> encodeAndPad(
  String text,
  Map<String, int> vocab, {
  int maxLen = 40,
}) {
  final tokens = tokenize(text);
  final ids = tokensToIds(tokens.take(maxLen).toList(), vocab);
  final padding = List<int>.filled(maxLen - ids.length, padId);
  return ids + padding;
}

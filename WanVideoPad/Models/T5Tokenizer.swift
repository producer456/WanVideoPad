// T5 Tokenizer wrapper
// Loads tokenizer.json and encodes text to token IDs + attention masks

import Foundation
import MLX

class T5TokenizerWrapper {
    private let vocabMap: [String: Int]
    private let padTokenId: Int

    init(tokenizerPath: URL) throws {
        let data = try Data(contentsOf: tokenizerPath)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        // Extract vocab from the model section
        var vocab: [String: Int] = [:]
        if let model = json["model"] as? [String: Any],
           let vocabEntries = model["vocab"] as? [String: Int] {
            vocab = vocabEntries
        }
        self.vocabMap = vocab
        self.padTokenId = vocab["<pad>"] ?? 0
    }

    func encode(_ text: String, maxLength: Int = 512) -> (inputIds: MLXArray, attentionMask: MLXArray) {
        // Simple whitespace tokenization fallback
        // For production, use the full SentencePiece/Unigram tokenizer
        var tokenIds: [Int32] = []

        // Split into words and look up in vocab
        let words = text.split(separator: " ")
        for word in words {
            let wordStr = String(word).lowercased()
            if let id = vocabMap["▁" + wordStr] {
                tokenIds.append(Int32(id))
            } else {
                // Character-level fallback
                for char in wordStr {
                    if let id = vocabMap[String(char)] {
                        tokenIds.append(Int32(id))
                    }
                }
            }
        }

        // Add EOS token
        if let eosId = vocabMap["</s>"] {
            tokenIds.append(Int32(eosId))
        }

        // Truncate
        if tokenIds.count > maxLength {
            tokenIds = Array(tokenIds.prefix(maxLength))
        }

        // Build attention mask
        var attentionMask = [Int32](repeating: 1, count: tokenIds.count)

        // Pad to maxLength
        let padLength = maxLength - tokenIds.count
        if padLength > 0 {
            tokenIds += [Int32](repeating: Int32(padTokenId), count: padLength)
            attentionMask += [Int32](repeating: 0, count: padLength)
        }

        return (
            inputIds: MLXArray(tokenIds).reshaped([1, maxLength]),
            attentionMask: MLXArray(attentionMask).reshaped([1, maxLength])
        )
    }
}

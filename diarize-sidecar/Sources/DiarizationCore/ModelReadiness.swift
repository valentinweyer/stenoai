import CoreML
import FluidAudio
import Foundation

public struct ModelReadinessStatus: Codable, Equatable, Sendable {
    public let ready: Bool
    public let cacheDirectory: String
    public let requiredModels: [String]
    public let missingModels: [String]

    enum CodingKeys: String, CodingKey {
        case ready
        case cacheDirectory = "cache_directory"
        case requiredModels = "required_models"
        case missingModels = "missing_models"
    }
}

public enum ModelReadiness {
    private static let modelDirectoryEnvironmentKey = "STENOAI_DIARIZE_MODEL_DIR"
    private static let userDataEnvironmentKey = "STENOAI_USER_DATA_DIR"

    /// Nemotron 3 bundles cache under this subdirectory of the cache root
    /// (FluidAudio's `Repo.nemotron3Diarization.folderName`).
    private static let nemotron3CacheFolder = "nemotron-3-diarization"

    public static let requiredModelRelativePaths: [String] =
        requiredModelRelativePaths(engine: .sortformer, nemotron3Config: nil)

    /// Model bundles and assets a given engine needs under the cache root.
    ///
    /// The Nemotron 3 engine needs its preset's CoreML bundle plus the
    /// root-level assets `Nemotron3Models.load` reads (`learnable_sil_emb.bin`,
    /// `pre_encode_proj_t.bin` for split-graph presets); both engines share
    /// the WeSpeaker/pyannote embedding models used for voiceprint centroids.
    public static func requiredModelRelativePaths(
        engine: DiarizationEngine,
        nemotron3Config: Nemotron3Config?
    ) -> [String] {
        let embeddingBundles = DiarizerModels.requiredModelNames
            .sorted()
            .map { "speaker-diarization/\($0)" }
        switch engine {
        case .sortformer:
            let sortformerBundles = [SortformerConfig.default, .highContextV2]
                .compactMap { ModelNames.Sortformer.bundle(for: $0) }
                .map { "sortformer/\($0)" }
            return sortformerBundles + embeddingBundles
        case .nemotron3:
            guard let config = nemotron3Config else {
                return []
            }
            let bundle = "\(nemotron3CacheFolder)/\(config.hubSubdirectory)/\(config.modelFileName)"
            var assets = ["\(nemotron3CacheFolder)/\(ModelNames.Nemotron3.silenceEmbeddingFile)"]
            if config.splitGraph {
                assets.append("\(nemotron3CacheFolder)/\(ModelNames.Nemotron3.preEncodeProjectionFile)")
            }
            return [bundle] + assets + embeddingBundles
        }
    }

    public static func cacheDirectory(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> URL {
        if let override = nonEmpty(environment[modelDirectoryEnvironmentKey]) {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        if let userData = nonEmpty(environment[userDataEnvironmentKey]) {
            return URL(fileURLWithPath: userData, isDirectory: true)
                .appendingPathComponent("models/speaker-diarization", isDirectory: true)
        }
        return homeDirectory
            .appendingPathComponent("Library/Application Support/stenoai", isDirectory: true)
            .appendingPathComponent("models/speaker-diarization", isDirectory: true)
    }

    public static func runtimeCacheDirectory(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser,
        engine: DiarizationEngine = .sortformer,
        nemotron3Config: Nemotron3Config? = nil
    ) -> URL {
        let preferred = cacheDirectory(environment: environment, homeDirectory: homeDirectory)
        if nonEmpty(environment[modelDirectoryEnvironmentKey]) != nil
            || nonEmpty(environment[userDataEnvironmentKey]) != nil
        {
            return preferred
        }
        if missingModelPaths(
            in: preferred, engine: engine, nemotron3Config: nemotron3Config
        ).isEmpty {
            return preferred
        }
        let legacy = homeDirectory
            .appendingPathComponent("Library/Application Support/FluidAudio/Models", isDirectory: true)
        return missingModelPaths(
            in: legacy, engine: engine, nemotron3Config: nemotron3Config
        ).isEmpty ? legacy : preferred
    }

    public static func status(
        cacheDirectory: URL? = nil,
        engine: DiarizationEngine = .sortformer,
        nemotron3Config: Nemotron3Config? = nil
    ) -> ModelReadinessStatus {
        let resolvedCacheDirectory =
            cacheDirectory ?? runtimeCacheDirectory(engine: engine, nemotron3Config: nemotron3Config)
        let missing = missingModelPaths(
            in: resolvedCacheDirectory, engine: engine, nemotron3Config: nemotron3Config
        )
        return ModelReadinessStatus(
            ready: missing.isEmpty,
            cacheDirectory: resolvedCacheDirectory.path,
            requiredModels: requiredModelRelativePaths(engine: engine, nemotron3Config: nemotron3Config),
            missingModels: missing
        )
    }

    private static func missingModelPaths(
        in cacheDirectory: URL,
        engine: DiarizationEngine,
        nemotron3Config: Nemotron3Config?
    ) -> [String] {
        requiredModelRelativePaths(engine: engine, nemotron3Config: nemotron3Config)
            .filter { relativePath in
                if relativePath.hasSuffix(".mlmodelc") {
                    return !isCompleteModelBundle(
                        cacheDirectory.appendingPathComponent(relativePath, isDirectory: true),
                        relativePath: relativePath
                    )
                }
                return !isCompleteAssetFile(cacheDirectory.appendingPathComponent(relativePath))
            }
    }

    public static func prepare(
        cacheDirectory: URL = cacheDirectory(),
        engine: DiarizationEngine = .sortformer,
        nemotron3Config: Nemotron3Config? = nil,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        progressHandler: ProgressHandler? = nil
    ) async throws -> ModelReadinessStatus {
        ModelHub.offlineMode = false
        try FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)

        switch engine {
        case .sortformer:
            _ = try await SortformerModels.loadFromHuggingFace(
                config: .default,
                cacheDirectory: cacheDirectory,
                computeUnits: computeUnits,
                progressHandler: progressHandler
            )
            _ = try await SortformerModels.loadFromHuggingFace(
                config: .highContextV2,
                cacheDirectory: cacheDirectory,
                computeUnits: computeUnits,
                progressHandler: progressHandler
            )
        case .nemotron3:
            guard let config = nemotron3Config else {
                throw CocoaError(
                    .fileReadCorruptFile,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "Nemotron 3 diarization requires a valid preset "
                            + "(STENOAI_DIARIZE_NEMOTRON_PRESET)"
                    ]
                )
            }
            _ = try await Nemotron3Models.loadFromHuggingFace(
                config: config,
                cacheDirectory: cacheDirectory,
                computeUnits: computeUnits,
                progressHandler: progressHandler
            )
        }
        _ = try await DiarizerModels.downloadIfNeeded(
            to: cacheDirectory.appendingPathComponent("speaker-diarization", isDirectory: true),
            configuration: MLModelConfigurationUtils.defaultConfiguration(computeUnits: computeUnits),
            progressHandler: progressHandler
        )

        let result = status(
            cacheDirectory: cacheDirectory, engine: engine, nemotron3Config: nemotron3Config
        )
        guard result.ready else {
            throw CocoaError(
                .fileReadCorruptFile,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Speaker diarization model setup completed with missing model bundles: "
                        + result.missingModels.joined(separator: ", ")
                ]
            )
        }
        return result
    }

    public static func enableOfflineOnly() {
        ModelHub.offlineMode = true
    }

    private static func isCompleteModelBundle(_ url: URL, relativePath: String) -> Bool {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            return false
        }
        return requiredArtifactRelativePaths(for: relativePath).allSatisfy { artifact in
            let path = url
                .appendingPathComponent(artifact, isDirectory: false)
                .resolvingSymlinksInPath()
                .path
            guard let attributes = try? FileManager.default.attributesOfItem(atPath: path),
                  attributes[.type] as? FileAttributeType == .typeRegular,
                  let size = attributes[.size] as? NSNumber else {
                return false
            }
            return size.intValue > 0
        }
    }

    /// Root-level assets (e.g. `learnable_sil_emb.bin`) are plain files, not
    /// compiled bundles — readiness only needs a non-empty regular file
    /// (symlinks allowed, same as bundle artifacts).
    private static func isCompleteAssetFile(_ url: URL) -> Bool {
        let path = url.resolvingSymlinksInPath().path
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: path),
              attributes[.type] as? FileAttributeType == .typeRegular,
              let size = attributes[.size] as? NSNumber else {
            return false
        }
        return size.intValue > 0
    }

    static func requiredArtifactRelativePaths(for relativePath: String) -> [String] {
        if relativePath.hasPrefix("sortformer/") {
            return [
                "coremldata.bin", "metadata.json",
                "model0/model.mil", "model0/weights/0-weight.bin",
                "model1/model.mil", "model1/weights/1-weight.bin",
            ]
        }
        if relativePath.hasPrefix("\(nemotron3CacheFolder)/") {
            // Nemotron 3 bundles ship without metadata.json (verified against the
            // Hugging Face repository layout: coremldata.bin, model.mil, weights/).
            return ["coremldata.bin", "model.mil", "weights/weight.bin"]
        }
        return ["coremldata.bin", "metadata.json", "model.mil", "weights/weight.bin"]
    }

    private static func nonEmpty(_ value: String?) -> String? {
        guard let value else { return nil }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}

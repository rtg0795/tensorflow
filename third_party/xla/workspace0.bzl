"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("@bazel_toolchains//repositories:repositories.bzl", bazel_toolchains_repositories = "repositories")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@build_bazel_apple_support//lib:repositories.bzl", "apple_support_dependencies")
load("@build_bazel_rules_apple//apple:repositories.bzl", "apple_rules_dependencies")
load("@build_bazel_rules_swift//swift:repositories.bzl", "swift_rules_dependencies")
load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
load("@com_google_benchmark//:bazel/benchmark_deps.bzl", "benchmark_deps")

def _tf_bind():
    """Bind targets for some external repositories"""
    ##############################################################################
    # BIND DEFINITIONS
    #
    # Please do not add bind() definitions unless we have no other choice.
    # If that ends up being the case, please leave a comment explaining
    # why we can't depend on the canonical build target.

    # Needed by Protobuf
    native.bind(
        name = "grpc_cpp_plugin",
        actual = "@com_github_grpc_grpc//src/compiler:grpc_cpp_plugin",
    )
    native.bind(
        name = "grpc_python_plugin",
        actual = "@com_github_grpc_grpc//src/compiler:grpc_python_plugin",
    )

    native.bind(
        name = "grpc_lib",
        actual = "@com_github_grpc_grpc//:grpc++",
    )

    native.bind(
        name = "grpc_lib_unsecure",
        actual = "@com_github_grpc_grpc//:grpc++_unsecure",
    )

    # Needed by Protobuf
    native.bind(
        name = "python_headers",
        actual = str(Label("//third_party/python_runtime:headers")),
    )

    # Needed by Protobuf
    native.bind(
        name = "six",
        actual = "@six_archive//:six",
    )

def workspace():
    http_archive(
        name = "inception_v1",
        build_file = "//:models.BUILD",
        sha256 = "7efe12a8363f09bc24d7b7a450304a15655a57a7751929b2c1593a71183bb105",
        urls = [
            "https://storage.googleapis.com/download.tensorflow.org/models/inception_v1.zip",
        ],
    )

    http_archive(
        name = "mobile_ssd",
        build_file = "//:models.BUILD",
        sha256 = "bddd81ea5c80a97adfac1c9f770e6f55cbafd7cce4d3bbe15fbeb041e6b8f3e8",
        urls = [
            "https://storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_android_export.zip",
        ],
    )

    http_archive(
        name = "mobile_multibox",
        build_file = "//:models.BUILD",
        sha256 = "859edcddf84dddb974c36c36cfc1f74555148e9c9213dedacf1d6b613ad52b96",
        urls = [
            "https://storage.googleapis.com/download.tensorflow.org/models/mobile_multibox_v1a.zip",
        ],
    )

    http_archive(
        name = "stylize",
        build_file = "//:models.BUILD",
        sha256 = "3d374a730aef330424a356a8d4f04d8a54277c425e274ecb7d9c83aa912c6bfa",
        urls = [
            "https://storage.googleapis.com/download.tensorflow.org/models/stylize_v1.zip",
        ],
    )

    http_archive(
        name = "speech_commands",
        build_file = "//:models.BUILD",
        sha256 = "c3ec4fea3158eb111f1d932336351edfe8bd515bb6e87aad4f25dbad0a600d0c",
        urls = [
            "https://storage.googleapis.com/download.tensorflow.org/models/speech_commands_v0.01.zip",
        ],
    )

    http_archive(
        name = "person_detect_data",
        sha256 = "170542270da256994ce24d1e357f6e84a54fdaf7d28ff2b74725a40b70b082cf",
        urls = [
            "https://storage.googleapis.com/download.tensorflow.org/data/tf_lite_micro_person_data_grayscale_2020_05_24.zip",
        ],
    )

    bazel_toolchains_repositories()

    # Apple rules for Bazel. https://github.com/bazelbuild/rules_apple.
    # Note: We add this to fix Kokoro builds.
    # The rules below call into `rules_proto` but the hash has changed and
    # Bazel refuses to continue. So, we add our own mirror.
    http_archive(
        name = "rules_proto",
        sha256 = "20b240eba17a36be4b0b22635aca63053913d5c1ee36e16be36499d167a2f533",
        strip_prefix = "rules_proto-11bf7c25e666dd7ddacbcd4d4c4a9de7a25175f8",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_proto/archive/11bf7c25e666dd7ddacbcd4d4c4a9de7a25175f8.tar.gz",
            "https://github.com/bazelbuild/rules_proto/archive/11bf7c25e666dd7ddacbcd4d4c4a9de7a25175f8.tar.gz",
        ],
    )

    http_archive(
        name = "rules_shell",
        sha256 = "bc61ef94facc78e20a645726f64756e5e285a045037c7a61f65af2941f4c25e1",
        strip_prefix = "rules_shell-0.4.1",
        url = "https://github.com/bazelbuild/rules_shell/releases/download/v0.4.1/rules_shell-v0.4.1.tar.gz",
    )

    # Now, finally use the rules
    apple_rules_dependencies()
    swift_rules_dependencies()
    apple_support_dependencies()

    # We only need `benchmark_deps` to be able to have bazel query to work and not complain about missing `@libpfm`.
    benchmark_deps()

    # Toolchains for ML projects hermetic builds.
    # Details: https://github.com/google-ml-infra/rules_ml_toolchain
    http_archive(
        name = "rules_ml_toolchain",
        sha256 = "562e0517f4e833afe0de7bb8da49f9adafcbca30a8259f118a65b4adf533b51f",
        strip_prefix = "rules_ml_toolchain-4995c0be587c6e173fe8cf8dc614f92011f7913d",
        urls = [
            "https://github.com/google-ml-infra/rules_ml_toolchain/archive/4995c0be587c6e173fe8cf8dc614f92011f7913d.tar.gz",
        ],
    )

    # If a target is bound twice, the later one wins, so we have to do tf bindings
    # at the end of the WORKSPACE file.
    _tf_bind()

    grpc_extra_deps()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
xla_workspace0 = workspace

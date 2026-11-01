pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "PocoPaw"
include(":app")

// Icons: Yoin Symbols (io.github.p2o51:yoin-symbols), always built from source: ~/code/yoin-symbols
// by default, or -PyoinSymbolsDir=<path>. Without the checkout the build stops here.
val yoinSymbolsDir = providers.gradleProperty("yoinSymbolsDir")
    .getOrElse("${System.getProperty("user.home")}/code/yoin-symbols")
val yoinSymbolsBuild = file("$yoinSymbolsDir/android")
check(yoinSymbolsBuild.isDirectory) {
    "Icons come from Yoin Symbols source: git clone https://github.com/p2o51/yoin-symbols " +
        "${yoinSymbolsBuild.parentFile.absolutePath}, or pass -PyoinSymbolsDir=<path>."
}
includeBuild(yoinSymbolsBuild)

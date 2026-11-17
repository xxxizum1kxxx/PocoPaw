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

// Icons: Yoin Symbols (io.github.p2o51:yoin-symbols), built from source when ~/code/yoin-symbols
// (or -PyoinSymbolsDir=<path>) is present, otherwise resolved from Maven Central.
val yoinSymbolsDir = providers.gradleProperty("yoinSymbolsDir")
    .getOrElse("${System.getProperty("user.home")}/code/yoin-symbols")
val yoinSymbolsBuild = file("$yoinSymbolsDir/android")
if (yoinSymbolsBuild.isDirectory) includeBuild(yoinSymbolsBuild)

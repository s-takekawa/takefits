APP_NAME = "Takefits"
APP_VERSION = "0.3.3"
APP_BUILD_SUFFIX = ""

# Keep the package version stable while making dev builds obvious in UI/CLI text.
APP_DISPLAY_VERSION = f"{APP_VERSION}-{APP_BUILD_SUFFIX}" if APP_BUILD_SUFFIX else APP_VERSION
APP_VERSION_TEXT = f"{APP_NAME} version {APP_DISPLAY_VERSION}"

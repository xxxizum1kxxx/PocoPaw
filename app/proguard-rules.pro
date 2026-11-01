# kotlinx.serialization: keep generated serializers for @Serializable classes (navigation routes).
-keepattributes *Annotation*, InnerClasses
-keepclassmembers class com.izum1k.pocopaw.** {
    *** Companion;
}
-keepclasseswithmembers class com.izum1k.pocopaw.** {
    kotlinx.serialization.KSerializer serializer(...);
}

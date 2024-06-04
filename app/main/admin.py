from django.contrib import admin
from .models import Signature, SignatureFeature
from .image_processing import process_image_and_extract_features
import json


class SignatureAdmin(admin.ModelAdmin):
    list_display = ["id", "upload_date", "get_html_image"]
    list_filter = ["upload_at"]


class SignatureFeatureAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "name",
        "contour_features_display",
        "geometric_features_display",
        "pixel_statistics_display",
        "texture_features_display",
        "frequency_features_display",
        "image",
    )

    def _get_formatted_json(self, json_data):
        if json_data:
            return json.dumps(json_data, indent=2)[:97] + (
                "..." if len(json.dumps(json_data, indent=2)) > 97 else ""
            )
        return "None"

    def contour_features_display(self, obj):
        return self._get_formatted_json(obj.contour_features)

    contour_features_display.short_description = "Контуры"

    def geometric_features_display(self, obj):
        return self._get_formatted_json(obj.geometric_features)

    geometric_features_display.short_description = "Геометрия"

    def pixel_statistics_display(self, obj):
        return self._get_formatted_json(obj.pixel_statistics)

    pixel_statistics_display.short_description = "Пиксели"

    def texture_features_display(self, obj):
        return self._get_formatted_json(obj.texture_features)

    texture_features_display.short_description = "Текстуры"

    def frequency_features_display(self, obj):
        return self._get_formatted_json(obj.frequency_features)

    frequency_features_display.short_description = "Частота"

    fields = ["image", "name"]

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        features = process_image_and_extract_features(obj.image.path)
        obj.contour_features = features.get("contour_features")
        obj.geometric_features = features.get("geometric_features")
        obj.pixel_statistics = features.get("pixel_statistics")
        obj.texture_features = features.get("texture_features")
        obj.frequency_features = features.get("frequency_features")
        obj.save()


admin.site.register(SignatureFeature, SignatureFeatureAdmin)
admin.site.register(Signature, SignatureAdmin)

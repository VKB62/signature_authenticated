from django.db import models
from django.contrib import admin
from django.utils.html import format_html
from django.utils import timezone


class Signature(models.Model):
    upload_at = models.DateTimeField(auto_now_add=True)
    image = models.ImageField("изображение", upload_to="signatures/")

    def __str__(self):
        return f"Signature(id={self.id})"

    @admin.display(description="Дата создания")
    def upload_date(self):
        if self.upload_at.date() == timezone.now().date():
            created_time = self.upload_at.time().strftime("%H:%M:%S")
            return format_html(
                '<span style="color: green; font-weight: bold;">Сегодня в {}</span>',
                created_time,
            )
        return self.upload_at.strftime("%d.%m.%y")

    @admin.display(description="фото")
    def get_html_image(self):
        if self.image:
            return format_html(
                '<img src="{url}" style="max-width: 80px; max-height: 80px;">',
                url=self.image.url,
            )

    class Meta:
        db_table = "signature"


class SignatureFeature(models.Model):
    contour_features = models.JSONField(null=True)
    geometric_features = models.JSONField(null=True)
    pixel_statistics = models.JSONField(null=True)
    texture_features = models.JSONField(null=True)
    frequency_features = models.JSONField(null=True)
    image = models.ImageField("изображение", upload_to="signatures_feature/")
    name = models.CharField(max_length=128, verbose_name="Имя пользователя")

    class Meta:
        db_table = "signature_feature"

    def __str__(self):
        return f"SignatureFeature(id={self.id})"

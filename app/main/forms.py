from django import forms
from .models import Signature


class SignatureForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["image"].widget.attrs["class"] = "form-control form-control-lg"

    class Meta:
        model = Signature
        fields = ["image"]

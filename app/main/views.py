from django.db.models import QuerySet
from django.shortcuts import render
from .forms import SignatureForm
from .models import SignatureFeature
from .image_processing import process_image_and_extract_features, compare_features
from typing import Any


def index(request):
    if request.method == "POST":
        form: SignatureForm = SignatureForm(request.POST, request.FILES)
        if form.is_valid():
            signature = form.save(commit=False)
            signature.save()

            uploaded_features: dict[str, Any] = process_image_and_extract_features(signature.image.path)
            signature_id: int | None
            similarity_score: float
            signature_id, similarity_score = compare_features(
                uploaded_features, SignatureFeature.objects.all()
            )

            if signature_id is not None:
                result_signature: QuerySet = SignatureFeature.objects.get(id=signature_id)
                return render(
                    request,
                    "main.html",
                    {
                        "result": {
                            "name": result_signature.name,
                            "proc": similarity_score,
                        },
                        "form": form,
                    },
                )
    else:
        form: SignatureForm = SignatureForm()

    return render(request, "main.html", {"form": form, "result": None})

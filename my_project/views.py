from django.http import JsonResponse
from django.shortcuts import render, redirect
from .models import *
from model import *


def index(request):
    return render(request, "index.html")


def predict(request):
    answer = [0, 0, 0, 0, 0, 0]
    name_catagorys = [
        "Computer Science",
        "Physics",
        "Mathematics",
        "Statistics",
        "Quantitative Biology",
        "Quantitative Finance",
    ]
    if request.method == "POST":
        text = request.POST.get("input")
        print(text)
        answer = runModel(text, 300)
    positions = [index for index, value in enumerate(answer) if value > 0.5]
    name_catagoryss = [name_catagorys[index] for index in positions]

    return JsonResponse({"answer": answer, "name_catagory": name_catagoryss})

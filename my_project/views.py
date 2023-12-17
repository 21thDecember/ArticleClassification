from django.http import JsonResponse
from django.shortcuts import render, redirect
from .models import *
from model import *


def index(request):
    return render(request, 'index.html')

def predict(request):
    answer = [0,0,0,0,0,0]
    if request.method == "POST":
        text = request.POST.get("input")
        print(text)
        answer = runModel(text, 300)
    return JsonResponse(
        {'answer': answer}
    )
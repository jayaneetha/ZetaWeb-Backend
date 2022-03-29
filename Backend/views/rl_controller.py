from django.http import HttpRequest, JsonResponse

from Backend.service.rl_service import store_file, process_audio, process_feedback


def post_upload(request: HttpRequest):
    file = request.FILES["audioFile"]
    file_url = store_file(file)
    audio_id, prediction = process_audio(file_url)

    context = {
        'audio_id': audio_id,
        'prediction': prediction
    }
    return JsonResponse(context)


def post_feedback(request: HttpRequest):
    audio_id = request.POST['audio_id']
    feedback = request.POST['feedback']
    process_feedback(audio_id, feedback)
    context = {
        'audio_id': audio_id,
    }
    return JsonResponse(context)

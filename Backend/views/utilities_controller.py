from django.http import HttpRequest, JsonResponse

from Backend.rl import STATE_STORE


def get_state_store(request: HttpRequest):
    context = {
        'UUIDs': list(STATE_STORE.keys())
    }
    return JsonResponse(context)

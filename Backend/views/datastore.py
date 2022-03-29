from django.http import HttpRequest, JsonResponse

from Backend.rl import DATASTORE


def get_datastore(request: HttpRequest):
    datastore = DATASTORE
    (mfcc, emotions, _), (_, _, _) = datastore.get_data()
    context = {
        'mfcc_shape': mfcc.shape
    }
    return JsonResponse(context)

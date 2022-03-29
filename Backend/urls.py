from django.urls import path

from .views.datastore import get_datastore
from .views.index import index
from .views.rl_controller import post_upload, post_feedback
from .views.utilities_controller import get_state_store

urlpatterns = [
    path('', index, name='index'),
    path('upload', post_upload),
    path('feedback', post_feedback),
    path('view', get_datastore),
    path('state_store', get_state_store)

]

from django.db import models


def upload_to(instance, filename):
    return 'images/{filename}'.format(filename=filename)


class UploadModel(models.Model):
    image = models.ImageField(upload_to=upload_to)

class info(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.TextField()
    infomation = models.TextField()
    class Meta:
        db_table = 'info'
# Generated by Django 2.2.2 on 2020-02-11 14:03

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ccrecapp', '0003_auto_20200211_1035'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='details',
            name='unnamed',
        ),
    ]
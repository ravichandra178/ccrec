# Generated by Django 2.2.2 on 2020-02-11 10:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ccrecapp', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='details',
            name='age',
            field=models.IntegerField(),
        ),
    ]

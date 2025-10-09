# Generated via `python manage.py makemigrations` (included for completeness)
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name='PredictionLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('input_text', models.TextField(help_text='Raw user input (pre-validation).')),
                ('cleaned_text', models.TextField(blank=True, help_text='Normalized text used for tokenization.')),
                ('seq_length_used', models.PositiveIntegerField(default=50)),
                ('generated_text', models.TextField(blank=True)),
                ('success', models.BooleanField(default=False)),
                ('error_message', models.TextField(blank=True)),
                ('latency_ms', models.PositiveIntegerField(default=0)),
                ('client_ip', models.GenericIPAddressField(blank=True, null=True)),
                ('user_agent', models.TextField(blank=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
            options={'ordering': ['-created_at']},
        ),
    ]

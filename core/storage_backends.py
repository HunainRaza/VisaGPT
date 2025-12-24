from django.core.files.storage import Storage
from supabase import create_client
from django.conf import settings

class SupabaseStorage(Storage):
    def __init__(self):
        self.client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_KEY
        )
        self.bucket = settings.SUPABASE_BUCKET
    
    def _save(self, name, content):
        self.client.storage.from_(self.bucket).upload(
            name,
            content.read()
        )
        return name
    
    def url(self, name):
        return f"{settings.SUPABASE_URL}/storage/v1/object/public/{self.bucket}/{name}"
    
    def exists(self, name):
        try:
            self.client.storage.from_(self.bucket).download(name)
            return True
        except:
            return False
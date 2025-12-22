from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.views.generic import TemplateView, ListView, DetailView, View
from django.contrib import messages
from .models import Document, DocumentValidation


@method_decorator(login_required, name='dispatch')
class DocumentUploadView(TemplateView):
    """Document upload view"""
    template_name = 'documents/upload.html'
    
    def post(self, request):
        doc_type = request.POST.get('document_type')
        file = request.FILES.get('file')
        
        if file:
            document = Document.objects.create(
                user=request.user,
                document_type=doc_type,
                file=file,
                original_filename=file.name,
                file_size=file.size
            )
            
            messages.success(request, f'Document "{file.name}" uploaded successfully!')
            
            # TODO: Trigger OCR processing
            
            return redirect('documents:list')
        else:
            messages.error(request, 'Please select a file to upload.')
            return self.render_to_response(self.get_context_data())


@method_decorator(login_required, name='dispatch')
class DocumentListView(ListView):
    """List all user documents"""
    model = Document
    template_name = 'documents/documents_list.html'
    context_object_name = 'documents'
    
    def get_queryset(self):
        return Document.objects.filter(user=self.request.user)


@method_decorator(login_required, name='dispatch')
class DocumentDetailView(DetailView):
    """View document details"""
    model = Document
    template_name = 'documents/details.html'
    context_object_name = 'document'
    
    def get_queryset(self):
        # Only allow users to view their own documents
        return Document.objects.filter(user=self.request.user)


@method_decorator(login_required, name='dispatch')
class DocumentDeleteView(View):
    """Delete document"""
    
    def post(self, request, pk):
        document = get_object_or_404(Document, pk=pk, user=request.user)
        filename = document.original_filename
        
        # Delete the file
        document.file.delete()
        document.delete()
        
        messages.success(request, f'Document "{filename}" deleted successfully!')
        return redirect('documents:list')


# Backward compatibility - keep old names as aliases
upload = DocumentUploadView.as_view()
document_list = DocumentListView.as_view()
document_detail = DocumentDetailView.as_view()
document_delete = DocumentDeleteView.as_view()
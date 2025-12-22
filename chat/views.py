from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.views.generic import TemplateView, View
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from .models import ChatConversation, ChatMessage
from .rag import VisaRAG
import json
import uuid

# Initialize RAG system
rag_system = VisaRAG()


class IndexView(TemplateView):
    """Landing page"""
    template_name = 'chat/index.html'


@method_decorator(login_required, name='dispatch')
class ChatView(TemplateView):
    """Main chat interface"""
    template_name = 'chat/chat.html'
    
    def get(self, request, *args, **kwargs):
        session_id = request.session.get('chat_session_id')
        
        if not session_id:
            session_id = str(uuid.uuid4())
            request.session['chat_session_id'] = session_id
            ChatConversation.objects.create(
                session_id=session_id,
                user=request.user
            )
        
        conversation = ChatConversation.objects.get(session_id=session_id)
        chat_messages = conversation.messages.all()
        
        context = self.get_context_data(**kwargs)
        context['chat_messages'] = chat_messages
        context['session_id'] = session_id
        
        return self.render_to_response(context)


@method_decorator(csrf_exempt, name='dispatch')
class ChatMessageView(View):
    """API endpoint for chat messages"""
    
    def post(self, request, *args, **kwargs):
        try:
            data = json.loads(request.body)
            user_message = data.get('message')
            session_id = request.session.get('chat_session_id')
            
            if not session_id:
                return JsonResponse({'error': 'No active session'}, status=400)
            
            conversation = ChatConversation.objects.get(session_id=session_id)
            
            # Save user message
            ChatMessage.objects.create(
                conversation=conversation,
                role='user',
                content=user_message
            )
            
            recent_messages = conversation.messages.order_by('-timestamp')[:5]
            history = [
                {'role': msg.role, 'content': msg.content}
                for msg in reversed(recent_messages)  # Reverse to get chronological order
            ]
            
            # Generate AI response using RAG
            ai_response = rag_system.generate_response(
                user_message, 
                history[:-1]  # Exclude current message
            )
            
            # Save AI response
            ChatMessage.objects.create(
                conversation=conversation,
                role='assistant',
                content=ai_response
            )
            
            return JsonResponse({
                'response': ai_response,
                'success': True
            })
            
        except Exception as e:
            return JsonResponse({
                'error': str(e),
                'success': False
            }, status=500)
    
    def get(self, request, *args, **kwargs):
        return JsonResponse({'error': 'Invalid method'}, status=405)


@method_decorator(login_required, name='dispatch')
class NewChatView(View):
    """Start a new chat conversation"""
    
    def get(self, request):
        # Clear current session
        if 'chat_session_id' in request.session:
            del request.session['chat_session_id']
        
        return redirect('chat:chat')
    
    def post(self, request):
        # Also handle POST
        if 'chat_session_id' in request.session:
            del request.session['chat_session_id']
        
        return redirect('chat:chat')


# Backward compatibility - keep old names as aliases
# index = IndexView.as_view()
# chat = ChatView.as_view()
# chat_message = ChatMessageView.as_view()
# new_chat = NewChatView.as_view()
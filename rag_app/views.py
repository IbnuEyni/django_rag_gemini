import os
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

def initialize_components(pdf_path):
    # Load the PDF data
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    
    # Initialize the text splitter with chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    
    # Split the loaded data into documents with overlap
    docs = text_splitter.split_documents(data)

    # Initialize embeddings with API key from settings
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=settings.GEMINI_API_KEY  # Use GEMINI_API_KEY here
        )
    )
    
    # Set up the retriever with similarity search
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # Initialize LLM with API key from settings
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        google_api_key=settings.GEMINI_API_KEY  # Use GEMINI_API_KEY here
    )

    return retriever, llm


@csrf_exempt
def chat_view(request):
    print("Entered chat_view function")  # Log entry into the function

    if request.method == 'POST':
        print("Handling POST request")  # Log that a POST request is being handled

        if 'pdf_file' in request.FILES:
            pdf_file = request.FILES['pdf_file']
            print("Received PDF file upload")  # Log that a PDF file was received

            # Save the PDF file to the 'data' directory
            fs = FileSystemStorage(location='data')
            file_name = fs.save(pdf_file.name, pdf_file)
            pdf_path = fs.url(file_name)
            print(f"PDF file saved at: {pdf_path}")  # Log where the PDF file was saved

            try:
                # Initialize components with the saved PDF path
                print("Initializing components with PDF file")  # Log that component initialization is starting
                retriever, llm = initialize_components(os.path.join('data', file_name))
                
                # Store retriever and llm in session
                request.session['retriever'] = retriever
                request.session['llm'] = llm
                print("Components initialized and stored in session successfully")  # Log successful initialization and storage

                return JsonResponse({"success": True, "message": "PDF uploaded successfully"})
            except Exception as e:
                print(f"Error initializing components: {e}")  # Log any errors during initialization
                return JsonResponse({"success": False, "error": "Error initializing components"})

        elif 'query' in request.POST:
            query = request.POST.get('query')
            if query:
                print(f"Received query: {query}")  # Log the received query

                try:
                    # Retrieve components from session
                    retriever = request.session.get('retriever')
                    llm = request.session.get('llm')
                    print("Retrieved components from session")  # Log that components were retrieved

                    if retriever is None or llm is None:
                        print("Error: Components not initialized or expired")  # Log if components are not available
                        return JsonResponse({"answer": "Error: Components not available. Please upload a PDF first."})

                    # Define the system prompt
                    system_prompt = (
                        "You are an assistant for question-answering tasks. "
                        "Use the following pieces of retrieved context to answer "
                        "the question. If you don't know the answer, say that you "
                        "don't know. Use three sentences maximum and keep the "
                        "answer concise."
                        "\n\n"
                        "{context}"
                    )

                    # Create the prompt template and the question-answer chain
                    prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", system_prompt),
                            ("human", "{input}"),
                        ]
                    )
                    print("Created ChatPromptTemplate")  # Log the creation of the prompt template

                    question_answer_chain = create_stuff_documents_chain(llm, prompt)
                    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                    print("Created RAG chain")  # Log the creation of the RAG chain

                    # Generate the response
                    response = rag_chain.invoke({"input": query})
                    print(f"Response generated: {response['answer']}")  # Log the generated response

                    return JsonResponse({"answer": response["answer"]})
                except Exception as e:
                    print(f"Error during query processing: {e}")  # Log any errors during query processing
                    return JsonResponse({"answer": "Error processing query. Please check the server logs."})

    print("Rendering chat.html")  # Log that the chat.html template is being rendered
    return render(request, 'chat.html')

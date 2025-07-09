# Bolt API - Memory Optimized

This Flask application provides email classification and response generation using AI models.

## Memory Optimizations

The application has been optimized to reduce memory usage on Render:

1. **Lazy Loading**: Models are loaded only when first requested
2. **CPU-only**: Forced CPU usage to avoid GPU memory allocation
3. **Memory-efficient settings**: Optimized PyTorch and Transformers configurations
4. **Single worker**: Gunicorn configured with 1 worker to minimize memory usage

## Deployment on Render

1. Connect your repository to Render
2. Set the following environment variables:
   - `GEMINI_KEY`: Your Google Gemini API key
3. The application will automatically use the optimized settings

## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your API key:
   ```
   GEMINI_KEY=your_api_key_here
   ```

3. Run the application:
   ```bash
   python Bolt.py
   ```

## API Endpoints

- `POST /analyze-label`: Analyze email and generate response
- `GET /health`: Health check endpoint
- `GET /`: Root endpoint

## Memory Usage Tips

- The application uses ~400-500MB of memory when models are loaded
- First request may take longer as models are loaded
- Subsequent requests will be faster
- Models are kept in memory for optimal performance
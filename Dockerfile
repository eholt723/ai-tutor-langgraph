FROM eholt723/ai-tutor-base:latest

WORKDIR /app

COPY . .

RUN chmod +x startup.sh

# HF Spaces requires port 7860
EXPOSE 7860

# Downloads models from HF Hub on first start, then launches uvicorn
CMD ["./startup.sh"]

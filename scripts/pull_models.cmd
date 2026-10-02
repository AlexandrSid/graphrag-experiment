@echo off
setlocal
echo Pulling gemma3:4b-it-q4_K_M
docker exec graphrag-ollama ollama pull gemma3:4b-it-q4_K_M
if errorlevel 1 exit /b 1
echo Pulling mxbai-embed-large
docker exec graphrag-ollama ollama pull mxbai-embed-large
if errorlevel 1 exit /b 1
echo Done. Check: http://localhost:11490/api/tags
exit /b 0

document.getElementById('generate-button').addEventListener('click', generateImage);
document.getElementById('prompt-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) { 
        e.preventDefault(); 
        generateImage();
    }
});

function generateImage() {
    const promptInput = document.getElementById('prompt-input');
    const prompt = promptInput.value.trim();

    if (prompt === "") {
        alert("Por favor, escribe una descripción.");
        return;
    }

    const loader = document.getElementById('loader');
    const button = document.getElementById('generate-button');
    const resultImage = document.getElementById('result-image');

    loader.style.display = 'block';
    resultImage.style.display = 'none';
    button.disabled = true;
    button.innerText = 'Generando...';

    // Se conecta al endpoint /generate-image de nuestro servidor de prueba
    fetch('/generate-image', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: prompt }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.imageUrl) {
            resultImage.src = data.imageUrl;
            resultImage.style.display = 'block';
            loader.style.display = 'none';
        } else if (data.error) {
            alert('Error: ' + data.error);
            loader.style.display = 'none';
        }
    })
    .catch((error) => {
        console.error('Error:', error);
        alert('Ocurrió un error de red. Revisa la consola de Flask.');
        loader.style.display = 'none';
    })
    .finally(() => {
        button.disabled = false;
        button.innerText = 'Generar';
    });
}
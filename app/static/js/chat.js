const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

// Fondo dinámico (pinceladas flotantes)
const canvas = document.getElementById("background");
const ctx = canvas.getContext("2d");
let w, h, particles = [];

function resize() {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
}
window.addEventListener("resize", resize);
resize();

function Particle() {
    this.x = Math.random() * w;
    this.y = Math.random() * h;
    this.size = 20 + Math.random() * 40;
    this.speedX = (Math.random() - 0.5) * 0.4;
    this.speedY = (Math.random() - 0.5) * 0.4;
    this.color = `hsla(${Math.random() * 50 + 30}, 50%, 70%, 0.2)`;
}

function animate() {
    ctx.fillStyle = "rgba(255, 250, 240, 0.05)";
    ctx.fillRect(0, 0, w, h);
    particles.forEach(p => {
        p.x += p.speedX;
        p.y += p.speedY;
        if (p.x < 0 || p.x > w || p.y < 0 || p.y > h) {
            p.x = Math.random() * w;
            p.y = Math.random() * h;
        }
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size / 4, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.fill();
    });
    requestAnimationFrame(animate);
}

for (let i = 0; i < 40; i++) particles.push(new Particle());
animate();

// Chat
function appendMessage(text, sender, imageUrl = null) {
    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message", sender);

    msgDiv.innerHTML = `<p>${text}</p>`;

    if (imageUrl) {
        const img = document.createElement("img");
        img.src = imageUrl;
        msgDiv.appendChild(img);

        // Botón de descarga
        const downloadBtn = document.createElement("a");
        downloadBtn.href = imageUrl;
        downloadBtn.download = `arte_${text.slice(0, 8)}.png`;
        downloadBtn.textContent = "⬇️ Descargar";
        downloadBtn.classList.add("download-btn");
        msgDiv.appendChild(downloadBtn);
    }

    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    appendMessage(message, "user");
    userInput.value = "";

    // Mostrar loader
    const loaderId = `loader-${Date.now()}`;
    const loaderDiv = document.createElement("div");
    loaderDiv.id = loaderId;
    loaderDiv.classList.add("message", "bot");
    loaderDiv.innerHTML = "<p>🎨 Generando imagen...</p>";
    chatBox.appendChild(loaderDiv);
    chatBox.scrollTop = chatBox.scrollHeight;

    let data;
    try {
        const res = await fetch("/api/predict/", {  // <-- endpoint de Gradio
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ data: [message] })  // <-- Gradio espera un array
        });
        data = await res.json();
    } catch (err) {
        console.error(err);
        loaderDiv.innerHTML = "❌ Error de conexión con el servidor.";
        return;
    }

    // Remover loader
    loaderDiv.remove();

    if (data.error) {
        appendMessage(`❌ ${data.error}`, "bot");
    } else {
        // Gradio devuelve la imagen en data[0]
        appendMessage("🖼️ Aquí tienes tu imagen generada.", "bot", data.data[0]);
    }
}


sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keydown", e => {
    if (e.key === "Enter") sendMessage();
});

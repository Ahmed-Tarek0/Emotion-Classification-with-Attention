// The backend runs on a different origin/port than this file, so we call it by full URL.
const API_BASE = "http://127.0.0.1:8000";

const EMOTION_COLORS = {
  joy: "#F4B400",
  sadness: "#4A7FE8",
  anger: "#E63946",
  fear: "#8338EC",
  surprise: "#FB8500",
  disgust: "#52B788",
};

const textInput = document.getElementById("text-input");
const button = document.getElementById("classify-btn");
const clearBtn = document.getElementById("clear-btn");
const resultBox = document.getElementById("result");
const errorBox = document.getElementById("error");
const barsBox = document.getElementById("bars");
const topEmotionBox = document.getElementById("top-emotion");
const emojiCircle = document.getElementById("emoji-circle");

async function classify() {
  const text = textInput.value.trim();
  errorBox.style.display = "none";
  if (!text) {
    errorBox.textContent = "Please enter some text first.";
    errorBox.style.display = "block";
    return;
  }

  button.disabled = true;
  button.classList.add("loading");

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }
    const data = await res.json();
    renderResult(data);
  } catch (e) {
    const message = e instanceof TypeError
      ? `Can't reach the API at ${API_BASE}. Is the backend running?`
      : e.message;
    errorBox.textContent = message || "Something went wrong.";
    errorBox.style.display = "block";
    resultBox.style.display = "none";
  } finally {
    button.disabled = false;
    button.classList.remove("loading");
  }
}

function renderResult(data) {
  document.getElementById("top-emoji").textContent = data.emoji;
  document.getElementById("top-label").textContent = data.label;
  document.getElementById("top-conf").textContent = (data.confidence * 100).toFixed(1) + "% confidence";

  topEmotionBox.style.borderColor = data.color;
  topEmotionBox.style.background = data.color + "12";
  emojiCircle.style.background = data.color + "22";

  const sorted = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
  barsBox.innerHTML = "";
  for (const [name, prob] of sorted) {
    const pct = (prob * 100).toFixed(1);
    const color = EMOTION_COLORS[name] || "#999999";
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <div class="name">${name}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%; background:${color}"></div></div>
      <div class="pct">${pct}%</div>
    `;
    barsBox.appendChild(row);
  }
  resultBox.style.display = "block";
}

function clearAll() {
  textInput.value = "";
  errorBox.style.display = "none";
  resultBox.style.display = "none";
  textInput.focus();
}

button.addEventListener("click", classify);
clearBtn.addEventListener("click", clearAll);
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) classify();
});

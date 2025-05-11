const SMALL_MODEL = "https://summit-mind-api-small-191769586244.us-east1.run.app/summarize";
const BASE_MODEL = "https://summit-mind-api-base-191769586244.us-east1.run.app/summarize";

async function summarize() {
    const dialogue = document.getElementById("dialogueInput").value.trim();
    if (!dialogue) return alert("Please paste or upload a dialogue!");

    // Reset output boxes
    document.getElementById("smallSummary").innerText = "⏳ Waiting for response...";
    document.getElementById("baseSummary").innerText = "⏳ Waiting for response...";
    document.getElementById("smallTime").innerText = "";
    document.getElementById("baseTime").innerText = "";

    // Show loading overlay
    const loadingElem = document.getElementById("loading");
    const loadingText = document.getElementById("loadingMessage");
    loadingElem.classList.remove("hidden");
    loadingElem.style.display = "flex";
    loadingElem.style.animation = "fadeIn 0.5s ease-in forwards";

    const messages = [
        "Summarizing... please wait.",
        "Warming up serverless backends (cold start)...",
        "Starting T5 models — this can take up to a minute.",
        "Still working on it... thanks for your patience!",
        "Tip: T5-Base takes longer but generates better results.",
        "First calls can take up to a minute as servers are asleep.",
        "After warming up the server running additional requests will show more production like response times.",
    ];

    let messageIndex = 0;
    loadingText.textContent = messages[messageIndex]; // Initialize with first message

    function cycleMessages() {
        messageIndex = (messageIndex + 1) % messages.length;
        loadingText.textContent = messages[messageIndex];
    }
    const messageInterval = setInterval(cycleMessages, 5000);

    document.getElementById("error").classList.add("hidden");
    document.getElementById("results").classList.remove("hidden");

    let responsesReceived = 0;

    const handleResponse = async (url, modelKey, outputId, timeId) => {
        const start = Date.now();
        try {
            const res = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ dialogue, t5_model: modelKey }),
            });
            const data = await res.json();
            const time = ((Date.now() - start) / 1000).toFixed(1);
            document.getElementById(outputId).innerText = data.summary || "⚠️ No summary returned.";
            document.getElementById(timeId).innerText = `(Took ${time}s)`;
        } catch (err) {
            console.error(`${modelKey} model error:`, err);
            document.getElementById(outputId).innerText = `❌ Error fetching ${modelKey} summary.`;
        } finally {
            responsesReceived++;
            if (responsesReceived === 1) {
                loadingElem.classList.add("hidden");
                loadingElem.style.display = "none";
                clearInterval(messageInterval);
            }

            if (responsesReceived < 2) {
                if (outputId === "smallSummary") {
                    document.getElementById("baseSummary").innerText = "⏳ Still summarizing...";
                } else {
                    document.getElementById("smallSummary").innerText = "⏳ Still summarizing...";
                }
            }
        }
    };

    // Start both calls independently
    handleResponse(SMALL_MODEL, "small", "smallSummary", "smallTime");
    handleResponse(BASE_MODEL, "base", "baseSummary", "baseTime");
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function (e) {
        document.getElementById('dialogueInput').value = e.target.result;
    };
    reader.readAsText(file);
}

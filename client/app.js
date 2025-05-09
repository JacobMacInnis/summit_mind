// const API_URL = "https://your-cloudrun-service-url/summarize"; // Replace this!
const API_URL = 'https://summit-mind-api-191769586244.us-east1.run.app/summarize';
async function summarize() {
    const dialogue = document.getElementById('dialogueInput').value.trim();
    if (!dialogue) return alert("Please paste or upload a dialogue!");

    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('results').classList.add('hidden');
    document.getElementById('error').classList.add('hidden');

    try {
        const [smallStart, baseStart] = [Date.now(), Date.now()];

        const [smallRes, baseRes] = await Promise.all([
            fetch(API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ dialogue, t5_model: "small" })
            }),
            fetch(API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ dialogue, t5_model: "base" })
            })
        ]);

        const [smallData, baseData] = await Promise.all([smallRes.json(), baseRes.json()]);
        const smallTime = (Date.now() - smallStart) / 1000;
        const baseTime = (Date.now() - baseStart) / 1000;

        document.getElementById('smallSummary').innerText = smallData.summary || "No summary generated.";
        document.getElementById('baseSummary').innerText = baseData.summary || "No summary generated.";

        document.getElementById('smallTime').innerText = `(Took ${smallTime.toFixed(1)}s)`;
        document.getElementById('baseTime').innerText = `(Took ${baseTime.toFixed(1)}s)`;

        document.getElementById('results').classList.remove('hidden');
    } catch (err) {
        console.error(err);
        document.getElementById('error').innerText = "Server error. Please try again later.";
        document.getElementById('error').classList.remove('hidden');
    } finally {
        document.getElementById('loading').classList.add('hidden');
    }
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

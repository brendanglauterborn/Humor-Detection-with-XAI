import { useState } from "react";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell
} from "recharts";

function App() {

  const [text, setText] = useState("");

  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState(null);

  const [chartData, setChartData] = useState([]);

  const [loading, setLoading] = useState(false);

  async function analyzeText() {

    if (!text.trim()) return;

    setLoading(true);

    try {

      // Prediction request
      const predictResponse = await fetch(
        "http://127.0.0.1:8000/predict",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            text: text
          })
        }
      );

      const predictData = await predictResponse.json();

      setPrediction(predictData.prediction);
      setConfidence(predictData.confidence);

      // XAI request
      const explainResponse = await fetch(
        "http://127.0.0.1:8000/explain",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            text: text
          })
        }
      );

      const explainData = await explainResponse.json();

      const formattedData = explainData.tokens.map((token, index) => ({
        token: token,
        score: explainData.scores[index]
      }));

      setChartData(formattedData);

    } catch (error) {

      console.error(error);

      setPrediction("Error connecting to backend.");

    }

    setLoading(false);
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        backgroundColor: "#2b2f36",
        color: "white",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        fontFamily: "Arial",
        padding: "30px"
      }}
    >

      <div
        style={{
          width: "900px",
          backgroundColor: "#1f2937",
          padding: "30px",
          borderRadius: "12px",
          boxShadow: "0px 0px 20px rgba(0,0,0,0.4)"
        }}
      >

      <h1
        style={{
          textAlign: "center",
          marginBottom: "20px",
          fontSize: "28px",
          fontWeight: "600",
          color: "#b0b7c3",
          letterSpacing: "1px"
        }}
      >
        Humor Detection with XAI
      </h1>

        <textarea
          rows="6"
          placeholder="Enter text..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          style={{
            width: "100%",
            padding: "12px",
            borderRadius: "8px",
            border: "none",
            resize: "none",
            fontSize: "16px"
          }}
        />

        <button
          onClick={analyzeText}
          style={{
            width: "100%",
            marginTop: "20px",
            padding: "12px",
            border: "none",
            borderRadius: "8px",
            backgroundColor: "#2563eb",
            color: "white",
            fontSize: "16px",
            cursor: "pointer"
          }}
        >
          {loading ? "Analyzing with XAI..." : "Analyze"}
        </button>

        {prediction && (
          <div
            style={{
              marginTop: "25px",
              textAlign: "center"
            }}
          >

            <h2>
              Prediction: {prediction}
            </h2>

          {confidence !== null && (
            <p>
              Confidence: {(confidence * 100).toFixed(2)}%
            </p>
          )}

          </div>
        )}

        {chartData.length > 0 && (

          <div
            style={{
              marginTop: "40px"
            }}
          >

            <h2
              style={{
                textAlign: "center",
                marginBottom: "20px"
              }}
            >
              Integrated Gradients Token Attribution
            </h2>

            <ResponsiveContainer width="100%" height={400}>

              <BarChart data={chartData}>

              <XAxis
                dataKey="token"
                angle={-45}
                textAnchor="end"
                interval={0}
                height={100}
                tick={{ fill: "white", fontSize: 14 }}
              />
                <YAxis />
                <Tooltip />

                <Bar dataKey="score">

                  {chartData.map((entry, index) => (

                    <Cell
                      key={`cell-${index}`}
                      fill={entry.score >= 0 ? "#4682B4" : "#F08080"}
                    />

                  ))}

                </Bar>

              </BarChart>

              <p
                style={{
                  textAlign: "center",
                  marginTop: "30px",
                  color: "#6b7280",
                  fontSize: "14px",
                  letterSpacing: "0.5px",
                  whiteSpace: "nowrap"
                }}
              >
                Developed by Brendan Lauterborn
              </p>

            </ResponsiveContainer>

          </div>

        )}

      </div>

    </div>
    
  );
}

export default App;
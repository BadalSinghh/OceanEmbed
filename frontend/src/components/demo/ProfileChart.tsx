"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface ProfileChartProps {
  depths: number[];
  prediction: (number | null)[];
  groundTruth: (number | null)[];
  modelName: string;
  lat?: number;
  lon?: number;
  date?: string;
}

export default function ProfileChart({
  depths,
  prediction,
  groundTruth,
  modelName,
  lat,
  lon,
  date,
}: ProfileChartProps) {
  // Recharts wants data as array of objects; depth on Y axis (reversed = surface up)
  const chartData = depths.map((d, i) => ({
    depth: d,
    gt: groundTruth[i],
    pred: prediction[i],
  }));

  return (
    <div>
      <div
        style={{
          fontSize: "0.78rem",
          color: "var(--col-text-tertiary)",
          marginBottom: 12,
        }}
      >
        {lat != null && lon != null
          ? `Vertical profile at ${lat.toFixed(2)}°N, ${lon.toFixed(2)}°E`
          : "Vertical profile"}
        {date ? ` — ${date}` : ""}
      </div>
      <ResponsiveContainer width="100%" height={460}>
        <LineChart
          data={chartData}
          layout="vertical"
          margin={{ top: 10, right: 20, left: 10, bottom: 20 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="rgba(40,90,140,0.15)"
          />
          <XAxis
            type="number"
            dataKey="gt"
            name="Temperature (°C)"
            label={{ value: "Temperature (°C)", position: "insideBottom", offset: -10, fill: "#6d8a9c", fontSize: 11 }}
            tick={{ fill: "#6d8a9c", fontSize: 11 }}
            domain={["auto", "auto"]}
          />
          <YAxis
            type="number"
            dataKey="depth"
            reversed
            name="Depth (m)"
            label={{ value: "Depth (m)", angle: -90, position: "insideLeft", fill: "#6d8a9c", fontSize: 11 }}
            tick={{ fill: "#6d8a9c", fontSize: 11 }}
            domain={[0, 1000]}
            ticks={[0, 50, 100, 200, 300, 500, 700, 1000]}
          />
          <Tooltip
            contentStyle={{
              background: "var(--col-surface-1)",
              border: "1px solid var(--col-border)",
              borderRadius: 6,
              fontSize: 12,
              color: "var(--col-text-primary)",
            }}
            formatter={(value: unknown, name: unknown) => [
              value != null && typeof value === "number" ? value.toFixed(3) + " °C" : "—",
              name === "gt" ? "GLORYS12 Ground Truth" : modelName,
            ]}
            labelFormatter={(label) => `Depth: ${label} m`}
          />
          <Legend
            wrapperStyle={{ fontSize: 12, color: "var(--col-text-secondary)" }}
            formatter={(value) =>
              value === "gt" ? "GLORYS12 Ground Truth" : modelName
            }
          />
          <Line
            type="monotone"
            dataKey="gt"
            stroke="#4ca87a"
            strokeWidth={2.5}
            dot={{ r: 3, fill: "#4ca87a" }}
            connectNulls={false}
            name="gt"
          />
          <Line
            type="monotone"
            dataKey="pred"
            stroke="#d97c45"
            strokeWidth={2}
            strokeDasharray="5 3"
            dot={{ r: 3, fill: "#d97c45" }}
            connectNulls={false}
            name="pred"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

const width = 700, height = 700, margin = 40;
const radius = Math.min(width, height) / 2 - margin;

const svg = d3.select("#donut")
  .append("svg")
  .attr("width", width)
  .attr("height", height)
  .append("g")
  .attr("transform", `translate(${width / 2},${height / 2})`);

d3.csv("flights.csv").then(data => {
  // Aggregate total passengers per year
  const yearlyTotals = Array.from(
    d3.rollup(data, v => d3.sum(v, d => +d.passengers), d => d.year),
    ([year, total]) => ({ year, total })
  );

  const grandTotal = d3.sum(yearlyTotals, d => d.total);
  const color = d3.scaleOrdinal()
    .domain(yearlyTotals.map(d => d.year))
    .range(d3.schemeCategory10);

  const pie = d3.pie()
    .value(d => d.total)
    .sort(null);

  const arc = d3.arc()
    .innerRadius(radius * 0.5)  // Donut hole radius
    .outerRadius(radius);

  const outerArc = d3.arc()
    .innerRadius(radius * 0.9)
    .outerRadius(radius * 0.9);

  const arcs = svg.selectAll(".slice")
    .data(pie(yearlyTotals))
    .join("g")
    .attr("class", "slice");

  arcs.append("path")
    .attr("d", arc)
    .attr("fill", d => color(d.data.year))
    .attr("stroke", "white")
    .style("stroke-width", "2px");

  // Add percentage share labels
  arcs.append("text")
    .attr("transform", d => `translate(${arc.centroid(d)})`)
    .attr("text-anchor", "middle")
    .text(d => {
      const share = ((d.data.total / grandTotal) * 100).toFixed(2);
      return `${d.data.year}: ${share}%`;
    });
});
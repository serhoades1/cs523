const width = 800, height = 800;

const svg = d3.select("#chart")
  .append("svg")
  .attr("width", width)
  .attr("height", height);

d3.csv("flights.csv").then(data => {
  // 1. Build hierarchy dataset: Root -> Year -> Month
  const nestedData = {
    name: "Total",
    children: Array.from(
      d3.group(data, d => d.year),
      ([year, months]) => ({
        name: year,
        children: months.map(m => ({
          name: m.month,
          value: +m.passengers
        }))
      })
    )
  };

  const root = d3.hierarchy(nestedData)
    .sum(d => d.value)
    .sort((a, b) => b.value - a.value);

  const pack = d3.pack()
    .size([width, height])
    .padding(5);

  pack(root);

  const node = svg.selectAll("g")
    .data(root.descendants())
    .join("g")
    .attr("transform", d => `translate(${d.x},${d.y})`);

  node.append("circle")
    .attr("r", d => d.r)
    .attr("class", d => d.children ? "node" : "node node--leaf");

  node.filter(d => !d.children)
    .append("text")
    .text(d => `${d.data.name}: ${d.data.value}`);
});
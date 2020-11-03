const threshold = 0.7

async function init() {
  const model = await toxicity.load(threshold)

  chrome.tabs.executeScript({
    code: 'window.getSelection().toString()',
  }, async(selection) => {
    const selected = selection[0]
    document.getElementById('input').innerHTML = selected

    const table = document.getElementById('predictions-table')
    await model.classify(selected).then((predictions) => {
      predictions.forEach((category) => {
        // Add the results to the table
        const row = table.insertRow(-1)
        const labelCell = row.insertCell(0)
        const categoryCell = row.insertCell(1)
        categoryCell.innerHTML = category.results[0].match === null ? '-' : category.results[0].match.toString()
        labelCell.innerHTML = category.label
      })
    })
  })
}

init()
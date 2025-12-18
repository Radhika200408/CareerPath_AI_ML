import { useState } from 'react'

export default function App() {
  const [skills, setSkills] = useState('python sql')
  const [results, setResults] = useState([])

  const analyze = async () => {
    const res = await fetch('http://localhost:8000/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ skills: skills.split(' ') })
    })
    setResults(await res.json())
  }

  return (
    <div style={{ padding: 20 }}>
      <h2>CareerPath AI Dashboard</h2>
      <input value={skills} onChange={e => setSkills(e.target.value)} />
      <button onClick={analyze}>Analyze</button>

      {results.map(r => (
        <div key={r.role}>
          <h3>{r.role} ({r.match_score}%)</h3>
          <p>Missing Skills: {r.missing_skills.join(', ')}</p>
          <ul>{r.resources.map(x => <li key={x}>{x}</li>)}</ul>
        </div>
      ))}
    </div>
  )
}

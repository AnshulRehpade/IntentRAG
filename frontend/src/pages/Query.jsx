import { useState } from 'react'
import { Search, Zap, ShieldCheck, AlertTriangle, Clock, Brain } from 'lucide-react'
import api from '../api'

const INTENT_COLORS = {
  factual: 'bg-blue-100 text-blue-800',
  person: 'bg-purple-100 text-purple-800',
  time: 'bg-amber-100 text-amber-800',
  location: 'bg-green-100 text-green-800',
  explanation: 'bg-orange-100 text-orange-800',
  other: 'bg-gray-100 text-gray-800',
}

export default function Query() {
  const [query, setQuery] = useState('')
  const [selfHeal, setSelfHeal] = useState(true)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!query.trim()) return
    setError('')
    setLoading(true)
    setResult(null)

    try {
      const { data } = await api.post('/query', {
        query: query.trim(),
        self_heal: selfHeal,
      })

      if (data.success) {
        setResult(data.data)
      } else {
        setError(data.error || 'Query failed')
      }
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Connection failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold text-gray-900 mb-6">Query Pipeline</h1>

      {/* Query Input */}
      <form onSubmit={handleSubmit} className="mb-8">
        <div className="flex gap-3">
          <div className="flex-1 relative">
            <Search className="absolute left-4 top-3.5 text-gray-400" size={18} />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask anything... (e.g., Who created Python?)"
              className="w-full pl-11 pr-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none text-sm"
            />
          </div>
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="px-6 py-3 bg-indigo-600 text-white rounded-xl font-medium hover:bg-indigo-700 transition-colors disabled:opacity-50 flex items-center gap-2"
          >
            {loading ? (
              <span className="animate-spin">⟳</span>
            ) : (
              <Zap size={18} />
            )}
            {loading ? 'Running...' : 'Query'}
          </button>
        </div>

        <label className="flex items-center gap-2 mt-3 text-sm text-gray-600 cursor-pointer">
          <input
            type="checkbox"
            checked={selfHeal}
            onChange={(e) => setSelfHeal(e.target.checked)}
            className="rounded text-indigo-600"
          />
          Self-healing enabled (auto-retry on hallucination)
        </label>
      </form>

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6 text-sm">
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-6">
          {/* Answer Card */}
          <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
            <div className="flex items-center gap-3 mb-4">
              <span className={`px-2.5 py-1 rounded-full text-xs font-medium ${INTENT_COLORS[result.intent] || INTENT_COLORS.other}`}>
                {result.intent}
              </span>
              <span className="text-xs text-gray-500">
                confidence: {(result.intent_confidence * 100).toFixed(1)}%
              </span>
              <span className="text-xs text-gray-400 ml-auto flex items-center gap-1">
                <Clock size={12} /> {result.latency_ms}ms
              </span>
            </div>

            <p className="text-gray-900 leading-relaxed">{result.answer}</p>
          </div>

          {/* Hallucination Check */}
          <div className={`rounded-xl border p-4 ${
            result.hallucination_check?.is_hallucinated
              ? 'bg-red-50 border-red-200'
              : 'bg-green-50 border-green-200'
          }`}>
            <div className="flex items-center gap-2 mb-2">
              {result.hallucination_check?.is_hallucinated ? (
                <AlertTriangle size={18} className="text-red-600" />
              ) : (
                <ShieldCheck size={18} className="text-green-600" />
              )}
              <span className="font-medium text-sm">
                {result.hallucination_check?.is_hallucinated ? 'Hallucination Detected' : 'Answer Verified'}
              </span>
              <span className="text-xs ml-auto text-gray-500">
                confidence: {((result.hallucination_check?.confidence_score || 0) * 100).toFixed(0)}%
                {' · '}verdict: {result.hallucination_check?.llm_verdict || 'N/A'}
              </span>
            </div>
            {result.hallucination_check?.details && (
              <p className="text-xs text-gray-600">{result.hallucination_check.details}</p>
            )}
          </div>

          {/* Self-Healing */}
          {result.healing && result.healing.attempts > 1 && (
            <div className="bg-indigo-50 border border-indigo-200 rounded-xl p-4">
              <div className="flex items-center gap-2 mb-2">
                <Brain size={18} className="text-indigo-600" />
                <span className="font-medium text-sm text-indigo-900">
                  Self-Healing: {result.healing.was_healed ? 'Fixed' : 'Best attempt'}
                </span>
                <span className="text-xs text-indigo-600 ml-auto">
                  {result.healing.attempts} attempts
                </span>
              </div>
              <p className="text-xs text-indigo-700">
                Strategies: {result.healing.strategies_used.join(' → ')}
              </p>
              {result.healing.original_answer && (
                <details className="mt-2">
                  <summary className="text-xs text-indigo-600 cursor-pointer">Show original (hallucinated) answer</summary>
                  <p className="text-xs text-gray-600 mt-1 p-2 bg-white rounded">
                    {result.healing.original_answer}
                  </p>
                </details>
              )}
            </div>
          )}

          {/* Retrieved Chunks */}
          {result.retrieved_chunks?.length > 0 && (
            <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
              <h3 className="font-medium text-sm text-gray-700 mb-3">
                Retrieved Chunks ({result.retrieved_chunks.length})
              </h3>
              <div className="space-y-3">
                {result.retrieved_chunks.map((chunk, i) => (
                  <div key={i} className="p-3 bg-gray-50 rounded-lg text-sm">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-medium text-gray-500">Chunk {i + 1}</span>
                      <span className="text-xs text-indigo-600">
                        relevance: {(chunk.relevance_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <p className="text-gray-700 text-xs leading-relaxed">{chunk.content}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Model Info */}
          <div className="flex gap-4 text-xs text-gray-500">
            <span>Model: {result.model}</span>
            {result.usage?.total_tokens && <span>Tokens: {result.usage.total_tokens}</span>}
            {result.trace_id && <span>Trace: {result.trace_id}</span>}
          </div>
        </div>
      )}
    </div>
  )
}

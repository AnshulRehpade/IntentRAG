import { useState } from 'react'
import { Upload, FileText, CheckCircle } from 'lucide-react'
import api from '../api'

const INTENT_OPTIONS = ['factual', 'person', 'time', 'location', 'explanation', 'other']

export default function Ingest() {
  const [file, setFile] = useState(null)
  const [intentCategory, setIntentCategory] = useState('factual')
  const [source, setSource] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file) return
    setError('')
    setLoading(true)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('intent_category', intentCategory)
      if (source) formData.append('source', source)

      const { data } = await api.post('/ingest', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })

      if (data.success) {
        setResult(data.data)
        setFile(null)
        setSource('')
      } else {
        setError(data.error || 'Ingestion failed')
      }
    } catch (err) {
      const msg = err.response?.data?.detail || err.response?.data?.error || err.message
      setError(msg || 'Connection failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold text-gray-900 mb-6">Ingest Documents</h1>

      <form onSubmit={handleSubmit} className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm space-y-5">
        {/* File Upload */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Document File</label>
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-indigo-400 transition-colors">
            <FileText size={32} className="mx-auto text-gray-400 mb-2" />
            <input
              type="file"
              accept=".txt,.md,.csv"
              onChange={(e) => setFile(e.target.files[0])}
              className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"
            />
            {file && (
              <p className="text-sm text-indigo-600 mt-2">{file.name} ({(file.size / 1024).toFixed(1)} KB)</p>
            )}
          </div>
        </div>

        {/* Intent Category */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Intent Category</label>
          <select
            value={intentCategory}
            onChange={(e) => setIntentCategory(e.target.value)}
            className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none text-sm"
          >
            {INTENT_OPTIONS.map((opt) => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>
          <p className="text-xs text-gray-500 mt-1">
            Which type of questions should this document help answer?
          </p>
        </div>

        {/* Source */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Source (optional)</label>
          <input
            type="text"
            value={source}
            onChange={(e) => setSource(e.target.value)}
            placeholder="e.g., company wiki, meeting notes"
            className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none text-sm"
          />
        </div>

        {/* Submit */}
        <button
          type="submit"
          disabled={loading || !file}
          className="w-full bg-indigo-600 text-white py-3 rounded-lg font-medium hover:bg-indigo-700 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
        >
          {loading ? (
            <span className="animate-spin">⟳</span>
          ) : (
            <Upload size={18} />
          )}
          {loading ? 'Ingesting...' : 'Ingest Document'}
        </button>
      </form>

      {/* Error */}
      {error && (
        <div className="mt-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
          {error}
        </div>
      )}

      {/* Success */}
      {result && (
        <div className="mt-4 bg-green-50 border border-green-200 rounded-xl p-5">
          <div className="flex items-center gap-2 mb-3">
            <CheckCircle size={18} className="text-green-600" />
            <span className="font-medium text-green-800 text-sm">Document Ingested</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-sm text-gray-700">
            <div>File: <span className="font-medium">{result.filename}</span></div>
            <div>Intent: <span className="font-medium">{result.intent_category}</span></div>
            <div>Chunks: <span className="font-medium">{result.chunks_created}</span></div>
            <div>Doc ID: <span className="font-mono text-xs">{result.doc_id?.slice(0, 8)}...</span></div>
          </div>
        </div>
      )}
    </div>
  )
}

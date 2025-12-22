import { useState } from 'react'

function App() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const sendImage = async (file: File) => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch('http://localhost:8000/detect', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Failed to process image')
      }

      const data = await response.json()
      setResult(data)
    } catch (err: any) {
      setError(err.message || 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file)

      const reader = new FileReader()
      reader.onloadend = () => {
        setPreview(reader.result as string)
      }
      reader.readAsDataURL(file)

      sendImage(file)
    }
  }

  const handleReset = () => {
    setSelectedImage(null)
    setPreview(null)
    setResult(null)
    setError(null)
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <div className="bg-white rounded-lg shadow-lg p-8 max-w-md w-full">
        <h1 className="text-2xl font-bold text-gray-800 mb-6 text-center">
          Upload an Image
        </h1>

        {!preview ? (
          <label className="flex flex-col items-center justify-center w-full h-64 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer hover:border-blue-500 hover:bg-gray-50 transition-colors">
            <div className="flex flex-col items-center justify-center pt-5 pb-6">
              <p className="mb-2 text-sm text-gray-500 font-semibold">
                Click to upload
              </p>
              <p className="text-xs text-gray-400">
                PNG, JPG, GIF up to 10MB
              </p>
            </div>
            <input
              type="file"
              className="hidden"
              accept="image/*"
              onChange={handleImageChange}
            />
          </label>
        ) : (
          <div className="space-y-4">
            <img
              src={preview}
              alt="Preview"
              className="w-full h-64 object-cover rounded-lg"
            />

            <div className="flex gap-2">
              <label className="flex-1 bg-blue-500 hover:bg-blue-600 text-white font-medium py-2 px-4 rounded cursor-pointer text-center">
                Change Image
                <input
                  type="file"
                  className="hidden"
                  accept="image/*"
                  onChange={handleImageChange}
                />
              </label>
              <button
                onClick={handleReset}
                className="flex-1 bg-gray-500 hover:bg-gray-600 text-white font-medium py-2 px-4 rounded"
              >
                Remove
              </button>
            </div>

            {/* Status */}
            {loading && (
              <p className="text-sm text-blue-500 text-center">
                Analyzing image…
              </p>
            )}

            {error && (
              <p className="text-sm text-red-500 text-center">
                {error}
              </p>
            )}

            {result && (
              <div className="bg-gray-50 p-4 rounded text-sm text-gray-700">
                <pre className="whitespace-pre-wrap">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default App

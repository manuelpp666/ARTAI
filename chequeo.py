import sys
try:
    import langchain
    import langchain.chains
    print("✅ LangChain encontrado:", langchain.__version__)
    print("📂 Ubicación:", langchain.__file__)
except ImportError as e:
    print("❌ ERROR:", e)

print("\n🐍 Python ejecutándose desde:", sys.executable)
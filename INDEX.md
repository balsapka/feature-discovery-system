# Feature Discovery System - Documentation Index

Welcome to the Feature Discovery System! This index will help you navigate the documentation.

## 🚀 Getting Started

**New to the project?** Start here:

1. **[README.md](README.md)** - Overview, installation, and basic usage
2. **[QUICKSTART.md](QUICKSTART.md)** - Quick reference guide for common tasks
3. **[examples/basic_example.py](examples/basic_example.py)** - Working code example

**Setup:**
```bash
./setup.sh                        # Run setup script
cp .env.example .env              # Copy environment template
# Edit .env and add your API key
python examples/basic_example.py  # Test the system
```

---

## 📚 Documentation Overview

### Core Documentation

| Document | Description | Who Should Read |
|----------|-------------|-----------------|
| [README.md](README.md) | Project overview, installation, basic usage | Everyone |
| [BUILD_SUMMARY.md](BUILD_SUMMARY.md) | What was built, architecture, verification | Team leads, stakeholders |
| [QUICKSTART.md](QUICKSTART.md) | Quick reference, common patterns | Daily users |
| [API_REFERENCE.md](API_REFERENCE.md) | Complete API documentation | Developers |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Architecture, design decisions, contributing | Contributors |

### Configuration Files

| File | Purpose |
|------|---------|
| [.env.example](.env.example) | Environment variables template |
| [requirements.txt](requirements.txt) | Python dependencies |
| [setup.sh](setup.sh) | Automated setup script |

---

## 🎯 Use Case Guide

### "I want to use the system"

1. Read: [README.md](README.md) - Installation
2. Read: [QUICKSTART.md](QUICKSTART.md) - Usage patterns
3. Try: [examples/basic_example.py](examples/basic_example.py)
4. Reference: [API_REFERENCE.md](API_REFERENCE.md) when needed

### "I want to understand the architecture"

1. Read: [BUILD_SUMMARY.md](BUILD_SUMMARY.md) - High-level overview
2. Read: [DEVELOPMENT.md](DEVELOPMENT.md) - Detailed architecture
3. Review: Source code in [src/](src/)

### "I want to contribute or extend the system"

1. Read: [DEVELOPMENT.md](DEVELOPMENT.md) - Design principles
2. Read: [API_REFERENCE.md](API_REFERENCE.md) - API details
3. Review: [tests/test_workflow.py](tests/test_workflow.py) - Testing patterns
4. Check: Extension guides in [DEVELOPMENT.md](DEVELOPMENT.md)

### "I need help troubleshooting"

1. Check: [QUICKSTART.md](QUICKSTART.md) - Troubleshooting section
2. Review: [API_REFERENCE.md](API_REFERENCE.md) - Error handling
3. Check: Test suite [tests/](tests/)

---

## 📂 Source Code Organization

### Main Components

```
src/
├── workflow.py              # LangGraph orchestration
├── agents/
│   ├── feature_generator.py # Feature generation logic
│   └── evaluator.py         # Rubric creation & evaluation
└── models/
    └── schemas.py           # Pydantic data models
```

**Where to look:**
- **Workflow logic**: [src/workflow.py](src/workflow.py)
- **Feature generation**: [src/agents/feature_generator.py](src/agents/feature_generator.py)
- **Evaluation & rubrics**: [src/agents/evaluator.py](src/agents/evaluator.py)
- **Data structures**: [src/models/schemas.py](src/models/schemas.py)

---

## 🔍 Topic Index

### Architecture & Design

- **LangGraph workflow**: [DEVELOPMENT.md](DEVELOPMENT.md#code-architecture)
- **Evaluator-Optimizer pattern**: [DEVELOPMENT.md](DEVELOPMENT.md#evaluator-optimizer-pattern)
- **Agent design**: [DEVELOPMENT.md](DEVELOPMENT.md#agent-design)
- **State management**: [DEVELOPMENT.md](DEVELOPMENT.md#state-management)
- **Workflow diagram**: [BUILD_SUMMARY.md](BUILD_SUMMARY.md#workflow-architecture)

### Usage & Examples

- **Basic usage**: [README.md](README.md#usage)
- **CLI usage**: [API_REFERENCE.md](API_REFERENCE.md#cli-reference)
- **Python API**: [API_REFERENCE.md](API_REFERENCE.md#core-classes)
- **Code examples**: [examples/](examples/)
- **Quick patterns**: [QUICKSTART.md](QUICKSTART.md#usage-examples)

### Configuration

- **Environment setup**: [README.md](README.md#configuration)
- **LLM providers**: [API_REFERENCE.md](API_REFERENCE.md#environment-variables)
- **Parameters**: [API_REFERENCE.md](API_REFERENCE.md#featurediscoveryworkflow)
- **.env file**: [.env.example](.env.example)

### Data Models

- **Feature**: [API_REFERENCE.md](API_REFERENCE.md#feature)
- **EvaluationRubric**: [API_REFERENCE.md](API_REFERENCE.md#evaluationrubric)
- **FeatureScore**: [API_REFERENCE.md](API_REFERENCE.md#featurescore)
- **All models**: [src/models/schemas.py](src/models/schemas.py)

### Testing

- **Test suite**: [tests/test_workflow.py](tests/test_workflow.py)
- **Testing strategy**: [DEVELOPMENT.md](DEVELOPMENT.md#testing-strategy)
- **Running tests**: [API_REFERENCE.md](API_REFERENCE.md#testing-reference)

### Extension Points

- **Phase 2 (Feature Store)**: [DEVELOPMENT.md](DEVELOPMENT.md#adding-phase-2-feature-store-search)
- **Phase 3 (Ranking)**: [DEVELOPMENT.md](DEVELOPMENT.md#adding-phase-3-ranking-agent)
- **Aggregated features**: [DEVELOPMENT.md](DEVELOPMENT.md#adding-aggregated-features)
- **Custom agents**: [DEVELOPMENT.md](DEVELOPMENT.md#agent-design)

---

## 📖 Reading Paths

### Path 1: Quick Start (15 minutes)

```
README.md (5 min)
  ↓
QUICKSTART.md (5 min)
  ↓
examples/basic_example.py (5 min)
  ↓
Start using!
```

### Path 2: Deep Dive (1 hour)

```
README.md (5 min)
  ↓
BUILD_SUMMARY.md (15 min)
  ↓
DEVELOPMENT.md (25 min)
  ↓
Source code review (15 min)
  ↓
Ready to contribute!
```

### Path 3: Integration (30 minutes)

```
README.md (5 min)
  ↓
QUICKSTART.md (10 min)
  ↓
API_REFERENCE.md (15 min)
  ↓
Ready to integrate!
```

---

## 🛠️ Common Tasks

### Install and Run

```bash
./setup.sh
source venv/bin/activate
python examples/basic_example.py
```

**Documentation**: [README.md](README.md#installation)

### Use CLI

```bash
python cli.py --problem "Your business problem"
```

**Documentation**: [API_REFERENCE.md](API_REFERENCE.md#cli-reference)

### Use Python API

```python
from src.workflow import FeatureDiscoveryWorkflow

workflow = FeatureDiscoveryWorkflow()
result = workflow.run("Your business problem")
```

**Documentation**: [API_REFERENCE.md](API_REFERENCE.md#featurediscoveryworkflow)

### Run Tests

```bash
pytest tests/
```

**Documentation**: [API_REFERENCE.md](API_REFERENCE.md#testing-reference)

### Visualize Workflow

```bash
python examples/visualize_workflow.py
```

**Documentation**: [examples/visualize_workflow.py](examples/visualize_workflow.py)

### Add New Agent

**Documentation**: [DEVELOPMENT.md](DEVELOPMENT.md#extending-the-system)

### Troubleshoot

**Documentation**: [QUICKSTART.md](QUICKSTART.md#troubleshooting)

---

## 📊 Project Status

**Phase 1**: ✅ Complete
- Feature Generator Agent
- Evaluator Agent
- LangGraph workflow
- CLI interface
- Documentation

**Phase 2**: 🔄 Planned (Feature Store)
- Qdrant integration
- Vector search
- Feature retrieval

**Phase 3**: 🔄 Planned (Ranking)
- Ranking agent
- Top-k selection
- Context-aware ranking

**Documentation**: [BUILD_SUMMARY.md](BUILD_SUMMARY.md#-what-was-built)

---

## 🤝 Contributing

Want to contribute? Read:

1. [DEVELOPMENT.md](DEVELOPMENT.md#contributing) - Guidelines
2. [API_REFERENCE.md](API_REFERENCE.md) - API details
3. [tests/test_workflow.py](tests/test_workflow.py) - Test patterns

---

## 📞 Support

**Having issues?**

1. Check [QUICKSTART.md](QUICKSTART.md) troubleshooting
2. Review [API_REFERENCE.md](API_REFERENCE.md) error handling
3. Look at [examples/](examples/) for working code

---

## 📝 Document Summaries

### README.md
- Project overview
- Installation instructions
- Basic usage
- Configuration
- Quick examples

### BUILD_SUMMARY.md
- What was built
- Architecture overview
- Verification checklist
- Future phases
- Setup instructions

### QUICKSTART.md
- Project structure
- Quick reference
- Usage examples
- Configuration options
- Troubleshooting

### API_REFERENCE.md
- Complete API documentation
- All classes and methods
- CLI reference
- Environment variables
- Error handling

### DEVELOPMENT.md
- Design decisions
- Architecture details
- Extension guides
- Best practices
- Contributing guidelines

---

## 🎯 Next Steps

1. ✅ Read [README.md](README.md)
2. ✅ Run [setup.sh](setup.sh)
3. ✅ Configure [.env](.env.example)
4. ✅ Try [examples/basic_example.py](examples/basic_example.py)
5. ✅ Explore [API_REFERENCE.md](API_REFERENCE.md) as needed
6. 🔄 Plan Phase 2 implementation

---

**Happy feature discovery!** 🚀

*Last updated: January 2026*

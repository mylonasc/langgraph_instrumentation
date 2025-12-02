from instrumentation import PerfettoTracer, PerfettoLogger, LangGraphInstrumentationHandler
from assets.demo_graph import build_board_graph
import logging

def main():
    # Setup Instrumentation
    console_logger = PerfettoLogger("BoardRoom")
    console_logger.logger.setLevel(logging.INFO)
    
    tracer = PerfettoTracer()
    handler = LangGraphInstrumentationHandler(tracer=tracer, logger=console_logger)
    app = build_board_graph()
    topic = "Should we replace all developers with AI immediately?"
    try:
        result = app.invoke(
            {"topic": topic},
            config={
                "callbacks": [handler],
                "configurable": {"thread_id": "board_meeting_01"},
                "metadata": {
                    "graph_name": "BoardOfDirectors",
                    "agent_name": "Chairperson"
                }
            }
        )

        for analysis in result["analyses"]:
            print(analysis)
        print("-" * 60)
        print(f"⚖️ Final Decision:\n{result['final_decision']}")

    except Exception as e:
        console_logger.log_error("Main", "App", str(e))

    print("\n" + "="*60)
    tracer.save("parallel_llm_trace.json")
    print("👉 Open https://ui.perfetto.dev to see the Fan-Out/Fan-In pattern.")

if __name__ == "__main__":
    main()
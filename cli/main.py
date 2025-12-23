"""
CLI Application Logic

Provides interactive command-line interface for the Codebase Intelligence system.
"""
import logging
from typing import Optional, List
from core import CodebaseIntelligence

logger = logging.getLogger(__name__)


async def interactive_session(system: CodebaseIntelligence) -> None:
    """
    Run an interactive Q&A session with the codebase intelligence system.
    
    Args:
        system: Initialized CodebaseIntelligence instance
    """
    if not system.is_initialized():
        logger.error("System is not initialized. Cannot start interactive session.")
        return

    logger.info("\n%s", "=" * 70)
    logger.info("💬 INTERACTIVE CODEBASE Q&A")
    logger.info("%s", "=" * 70)
    logger.info("Commands: 'exit' to quit, 'save' to export audit log\n")

    while True:
        try:
            step = input("🔄 Step (🤔 query, 👋 exit, ✅ save, 📝 get_conversation_history, 🗂️ list_conversations, 🔍 get_conversation_state, 🧹 clear_conversation, 🗒️ get_conversation_summary): ").strip().lower()

            if step == "query":
                conversation_id = input("🆔 Conversation ID (or press Enter for 'default'): ").strip()
                user_input = input("\n🤔 Your question: ").strip()
                if not user_input:
                    continue
                result = await system.query(user_input, conversation_id or "default")
                logger.info('\n\n💡 %s', result)
                continue

            if step == 'exit':
                logger.info("\n👋 Goodbye!")
                break

            if step == 'save':
                try:
                    system.export_change_log()
                    logger.info("✅ Change log exported successfully!")
                except Exception as e:
                    logger.error(f"Failed to export change log: {e}")
                continue

            if step == "get_conversation_history":
                conversation_id = input("🆔 Conversation ID (or press Enter for 'default'): ").strip()
                history = await system.get_conversation_history(conversation_id or "default")
                logger.info("\n📝 Conversation History:\n%s", history)
                continue
                
            if step == "list_conversations":
                conversations = await system.list_conversations()
                logger.info("\n🗂️  Conversations:\n%s", conversations)
                continue
            
            if step == "get_conversation_state":
                conversation_id = input("🆔 Conversation ID (or press Enter for 'default'): ").strip()
                state = await system.get_conversation_state(conversation_id or "default")
                logger.info("\n🔍 Conversation State:\n%s", state)
                continue
                
            if step == "clear_conversation":
                conversation_id = input("🆔 Conversation ID (or press Enter for 'default'): ").strip()
                await system.clear_conversation(conversation_id or "default")
                logger.info("✅ Conversation cleared.")
                continue
            
            if step == "get_conversation_summary":
                conversation_id = input("🆔 Conversation ID (or press Enter for 'default'): ").strip()
                summary = await system.get_conversation_summary(conversation_id or "default")
                logger.info("\n🗒️  Conversation Summary:\n%s", summary)
                continue

            logger.warning("❓ Unknown command: %s", step)
        except KeyboardInterrupt:
            logger.info("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.exception("❌ Error during query: %s", e)


async def run_cli(
    force_reload: bool = False,
    skip_embeddings: bool = False,
    include_extensions: Optional[List[str]] = None,
    model_name: Optional[str] = None,
    interactive: bool = True
) -> CodebaseIntelligence:
    """
    Run the CLI application.
    
    Args:
        repo_path: Path to repository (defaults to auto-detected projects root)
        persist_dir: Directory for vector store persistence
        force_reload: Force rebuild of vector store
        skip_embeddings: Skip embedding creation
        include_extensions: File extensions to include
        model_name: LLM model to use
        interactive: Whether to start interactive session
        
    Returns:
        Initialized CodebaseIntelligence instance
    """
    logger.info("=" * 70)
    logger.info("🚀 CODEBASE INTELLIGENCE CLI")
    logger.info("=" * 70)

    # Initialize the system
    logger.info("⚙️  Initializing system...")
    system = CodebaseIntelligence(include_extensions=include_extensions, model_name=model_name)

    try:
        await system.initialize(
            force_reload=force_reload,
            skip_embeddings=skip_embeddings
        )
    except ValueError as e:
        logger.error(str(e))
        return system
    
    # Start interactive session if requested and system is ready
    if interactive and not skip_embeddings and system.is_initialized():
        try:
            await interactive_session(system)
        except Exception as e:
            logger.error(f"Interactive session ended: {e}")
    
    return system


async def main() -> None:
    """Default CLI entry point with standard configuration."""
    await run_cli(
        force_reload=False,
        skip_embeddings=False,
        interactive=True
    )

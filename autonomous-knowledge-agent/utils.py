# reset_udahub.py
import os
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph


Base = declarative_base()

def reset_db(db_path: str, echo: bool = True):
    """Drops the existing udahub.db file and recreates all tables."""

    # Remove the file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"✅ Removed existing {db_path}")

    # Create a new engine and recreate tables
    engine = create_engine(f"sqlite:///{db_path}", echo=echo)
    Base.metadata.create_all(engine)
    print(f"✅ Recreated {db_path} with fresh schema")


@contextmanager
def get_session(engine: Engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def model_to_dict(instance):
    """Convert a SQLAlchemy model instance to a dictionary."""
    return {
        column.name: getattr(instance, column.name)
        for column in instance.__table__.columns
    }

def chat_interface(
    agent: CompiledStateGraph,
    ticket_id: str,
    external_user_id: str | None = None,
    account_id: str = "cultpass",
    show_trace: bool = True,
):
    """Interactive support chat.

    ``ticket_id`` is used as the LangGraph ``thread_id`` (short-term memory).
    ``external_user_id`` is the CultPass customer id (e.g. ``"f556c0"``); it is
    only needed for a new ticket - existing tickets already know their customer.
    Type ``quit``/``exit``/``q`` to leave.
    """
    config = {"configurable": {"thread_id": ticket_id}}
    first_turn = True
    print(f"UDA-Hub support chat | ticket={ticket_id} | customer={external_user_id or 'from ticket / guest'}")
    while True:
        user_input = input("User: ").strip()
        if not user_input:
            continue
        print("User:", user_input)
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Assistant: Goodbye!")
            break
        trigger = {"messages": [HumanMessage(content=user_input)]}
        if first_turn and external_user_id:
            trigger.update(external_user_id=external_user_id, account_id=account_id)
        result = agent.invoke(input=trigger, config=config)
        reply = result["messages"][-1]
        print(f"Assistant ({getattr(reply, 'name', None) or 'ai'}):", reply.content)
        if show_trace:
            cls = result.get("classification") or {}
            print(
                f"   [category={cls.get('category')} | route={result.get('route')} | "
                f"ticket_status={result.get('ticket_status')}]"
            )
        first_turn = False

from dotenv import load_dotenv
import urllib.parse

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import MemorySaver

#Read ENV
load_dotenv()
username = "postgres"
password = "admin@1234"
host = "localhost"
port = "5432"
database = "my_tasks"
#Model
groq_llm_model = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    streaming=True
    # other params...
)


#DB
encoded_password = urllib.parse.quote_plus(password)
db_uri = f"postgresql+psycopg2://{username}:{encoded_password}@{host}:{port}/{database}"
db = SQLDatabase.from_uri(db_uri)
#print(db)


#TASKS TABLE QUERY
tasks_table = """
                CREATE TABLE IF NOT EXISTS TASKS (
                    ID INTEGER PRIMARY KEY,
                    TITLE TEXT NOT NULL,
                    DESCRIPTION TEXT,
                    STATUS TEXT CHECK (STATUS IN ('pending', 'in-progress', 'completed')) DEFAULT 'pending',
                    CREATED_TIME TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
              """
db.run(tasks_table)


#TOOLKIT
toolkit = SQLDatabaseToolkit(db=db,llm = groq_llm_model)
tools = toolkit.get_tools()

# for tool in tools:
#     print(tool.name)

#MEMORY
memory = MemorySaver()
system_prompt = """
You are a task management assistant that interacts with a SQL database containing a 'tasks' table.

TASK RULES:
1. Limit SELECT queries to 10 results max with ORDER BY created_time DESC
2. After CREATE/UPDATE/DELETE, confirm with SELECT query
3. If the user requests a list of tasks, present the output in a structured table format to ensure a clean and organized display in the browser."

CRUD OPERATIONS:
    CREATE: INSERT INTO tasks(title, description, status)
    READ: SELECT * FROM tasks WHERE ... LIMIT 10
    UPDATE: UPDATE tasks SET status=? WHERE id=? OR title=?
    DELETE: DELETE FROM tasks WHERE id=? OR title=?

Table schema: id, title, description, status(pending/in_progress/completed), created_time.
"""
agent = create_agent(
        model=groq_llm_model,
        tools=tools,
        checkpointer=memory,
        system_prompt=system_prompt
        )


thread_config = {"configurable": {"thread_id": "1"}}
while True:
    query = input("User : ")
    response = agent.invoke(
        {"messages":[{"role":"ai","content":query}]},
            thread_config
        )

    result = response["messages"][-1].content
    print(f"AI -> {result}")
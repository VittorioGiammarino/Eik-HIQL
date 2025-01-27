# pointmaze-medium-navigate-v0 (PI-HIQL)
python main.py --env_name=pointmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/pi_hiql.py 

# pointmaze-large-navigate-v0 (PI-HIQL)
python main.py --env_name=pointmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/pi_hiql.py 

# pointmaze-giant-navigate-v0 (PI-HIQL)
python main.py --env_name=pointmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/pi_hiql.py --agent.discount=0.995 

# pointmaze-teleport-navigate-v0 (PI-HIQL)
python main.py --env_name=pointmaze-teleport-navigate-v0 --eval_episodes=50 --agent=agents/pi_hiql.py

# pointmaze-medium-stitch-v0 (PI-HIQL)
python main.py --env_name=pointmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/pi_hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5

# pointmaze-large-stitch-v0 (PI-HIQL)
python main.py --env_name=pointmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/pi_hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5

# pointmaze-giant-stitch-v0 (PI-HIQL)
python main.py --env_name=pointmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/pi_hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.discount=0.995 

# pointmaze-teleport-stitch-v0 (PI-HIQL)
python main.py --env_name=pointmaze-teleport-stitch-v0 --eval_episodes=50 --agent=agents/pi_hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5

# antmaze-medium-navigate-v0 (PI-HIQL)
python main.py --env_name=antmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/pi_hiql.py

# antmaze-large-navigate-v0 (PI-HIQL)
python main.py --env_name=antmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/pi_hiql.py

# antmaze-giant-navigate-v0 (PI-HIQL)
python main.py --env_name=antmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/pi_hiql.py --agent.discount=0.995 

# antmaze-teleport-navigate-v0 (PI-HIQL)
python main.py --env_name=antmaze-teleport-navigate-v0 --eval_episodes=50 --agent=agents/pi_hiql.py

# antmaze-medium-stitch-v0 (PI-HIQL)
python main.py --env_name=antmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/pi_hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 

# antmaze-large-stitch-v0 (PI-HIQL)
python main.py --env_name=antmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/pi_hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 

# antmaze-giant-stitch-v0 (PI-HIQL)
python main.py --env_name=antmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/pi_hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.discount=0.995 

# antmaze-teleport-stitch-v0 (PI-HIQL)
python main.py --env_name=antmaze-teleport-stitch-v0 --eval_episodes=50 --agent=agents/pi_hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 

# antmaze-medium-explore-v0 (PI-HIQL)
python main.py --env_name=antmaze-medium-explore-v0 --eval_episodes=50 --agent=agents/pi_hiql.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.high_alpha=10.0 --agent.low_alpha=10.0

# antmaze-large-explore-v0 (PI-HIQL)
python main.py --env_name=antmaze-large-explore-v0 --eval_episodes=50 --agent=agents/pi_hiql.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.high_alpha=10.0 --agent.low_alpha=10.0

# antmaze-teleport-explore-v0 (PI-HIQL)
python main.py --env_name=antmaze-teleport-explore-v0 --eval_episodes=50 --agent=agents/pi_hiql.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.high_alpha=10.0 --agent.low_alpha=10.0

# humanoidmaze-medium-navigate-v0 (PI-HIQL)
python main.py --env_name=humanoidmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/pi_hiql.py --agent.discount=0.995 --agent.subgoal_steps=100

# humanoidmaze-large-navigate-v0 (PI-HIQL)
python main.py --env_name=humanoidmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/pi_hiql.py --agent.discount=0.995 --agent.subgoal_steps=100

# humanoidmaze-giant-navigate-v0 (PI-HIQL)
python main.py --env_name=humanoidmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/pi_hiql.py --agent.discount=0.995 --agent.subgoal_steps=100

# humanoidmaze-medium-stitch-v0 (PI-HIQL)
python main.py --env_name=humanoidmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/pi_hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.discount=0.995 --agent.subgoal_steps=100

# humanoidmaze-large-stitch-v0 (PI-HIQL)
python main.py --env_name=humanoidmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/pi_hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.discount=0.995 --agent.subgoal_steps=100

# humanoidmaze-giant-stitch-v0 (PI-HIQL)
python main.py --env_name=humanoidmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/pi_hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.discount=0.995 --agent.subgoal_steps=100

# antsoccer-arena-navigate-v0 (PI-HIQL)
python main.py --env_name=antsoccer-arena-navigate-v0 --eval_episodes=50 --agent=agents/pi_hiql.py

# antsoccer-medium-navigate-v0 (PI-HIQL)
python main.py --env_name=antsoccer-medium-navigate-v0 --eval_episodes=50 --agent=agents/pi_hiql.py

# antsoccer-arena-stitch-v0 (PI-HIQL)
python main.py --env_name=antsoccer-arena-stitch-v0 --eval_episodes=50 --agent=agents/pi_hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5

# antsoccer-medium-stitch-v0 (PI-HIQL)
python main.py --env_name=antsoccer-medium-stitch-v0 --eval_episodes=50 --agent=agents/pi_hiql.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5

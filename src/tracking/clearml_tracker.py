from clearml import Task


def init_task(topic: str) -> Task:
    """
    Initializes a ClearML experiment/task for one pipeline run.

    Args:
        topic (str): User-selected news topic

    Returns:
        Task: Active ClearML task object
    """
    task = Task.init(
        project_name="Perspective-Aware News Summarization",
        task_name=f"Run: {topic}",
        task_type=Task.TaskTypes.inference,
        reuse_last_task_id=False,
    )

    return task


def log_pipeline_config(task: Task, topic: str, page_size: int, max_articles: int):
    """
    Logs pipeline configuration parameters to ClearML.

    This helps make each experiment reproducible.

    Args:
        task (Task): Active ClearML task
        topic (str): News topic/query
        page_size (int): Number of articles requested per source
        max_articles (int): Maximum articles used downstream
    """
    task.connect({
        "topic":        topic,
        "page_size":    page_size,
        "max_articles": max_articles,
    }, name="Pipeline Config")


def log_fetch_metrics(task: Task, fetch_data: dict):
    """
    Logs article fetching metrics.

    Args:
        task (Task): Active ClearML task
        fetch_data (dict): Metadata returned from the fetching stage
    """
    logger = task.get_logger()

    # Track how many articles are collected and retained
    logger.report_scalar("Fetching", "total_fetched",    fetch_data.get("total_fetched", 0),   0)
    logger.report_scalar("Fetching", "after_relevance",  fetch_data.get("after_relevance", 0), 0)
    
    # Track contribution from each news source/API
    logger.report_scalar("Fetching", "newsapi_count",    fetch_data.get("newsapi_count", 0),   0)
    logger.report_scalar("Fetching", "guardian_count",   fetch_data.get("guardian_count", 0),  0)
    logger.report_scalar("Fetching", "gnews_count",      fetch_data.get("gnews_count", 0),     0)


def log_article_metrics(task: Task, source_summaries: list[dict]):
    """
    Logs relevance statistics and article-level metadata.

    Args:
        task (Task): Active ClearML task
        source_summaries (list[dict]): Articles/summaries used in the pipeline
    """
    logger = task.get_logger()

    # Extract relevance scores for aggregate statistics
    scores = [item.get("relevance_score", 0) for item in source_summaries]
    
    if scores:
        logger.report_scalar("Articles", "avg_relevance_score", round(sum(scores) / len(scores), 3), 0)
        logger.report_scalar("Articles", "max_relevance_score", max(scores), 0)
        logger.report_scalar("Articles", "min_relevance_score", min(scores), 0)
        logger.report_scalar("Articles", "total_articles_used", len(scores),  0)

    # Log source article details as formatted text for easy inspection
    lines = ["Source Articles:\n"]
    lines.append(f"{'#':<4} {'Source':<30} {'Score':<8} {'API':<12} Title")
    lines.append("-" * 90)


    for i, item in enumerate(source_summaries, 1):
        lines.append(
            f"{i:<4} {item.get('source',''):<30} "
            f"{item.get('relevance_score',0):<8.3f} "
            f"{item.get('api',''):<12} "
            f"{item.get('title','')[:50]}"
        )


    logger.report_text("\n".join(lines))


def log_comparison_metrics(task: Task, comparison_result: dict):
    """
    Logs comparison-stage metrics such as:
    - Perspective diversity
    - Common themes
    - Similar article pairs
    - Perspective distribution

    Args:
        task (Task): Active ClearML task
        comparison_result (dict): Output from comparison module
    """
    logger = task.get_logger()

    perspectives  = comparison_result.get("perspectives", [])
    common_themes = comparison_result.get("common_themes", [])
    similar_pairs = comparison_result.get("similar_pairs", [])
    diversity     = comparison_result.get("perspective_diversity", 0)

    # High-level comparison metrics
    logger.report_scalar("Comparison", "perspective_diversity",   diversity,          0)
    logger.report_scalar("Comparison", "similar_pairs_found",     len(similar_pairs), 0)
    logger.report_scalar("Comparison", "common_themes_extracted", len(common_themes), 0)

    # Count how many articles belong to each detected perspective
    perspective_counts = {}


    for p in perspectives:
        ptype = p.get("perspective", "General")
        perspective_counts[ptype] = perspective_counts.get(ptype, 0) + 1

    # Log perspective distribution as scalar metrics
    for ptype, count in perspective_counts.items():
        logger.report_scalar("Perspectives", ptype, count, 0)

    # Log extracted themes as readable text
    if common_themes:
        logger.report_text(f"Common themes: {', '.join(common_themes)}")

    # Log source-level perspective breakdown
    if perspectives:
        lines = ["Perspective Analysis:\n"]
        lines.append(f"{'Source':<30} {'Perspective':<15} Title")
        lines.append("-" * 80)

        for p in perspectives:
            lines.append(
                f"{p.get('source',''):<30} "
                f"{p.get('perspective',''):<15} "
                f"{p.get('title','')[:40]}"
            )

        logger.report_text("\n".join(lines))


def log_final_summary(task: Task, neutral_summary: str, topic: str):
    """
    Logs the final generated neutral summary.

    Args:
        task (Task): Active ClearML task
        neutral_summary (str): Final generated output
        topic (str): News topic/query
    """
    logger = task.get_logger()

    # Store final human-readable result in ClearML
    logger.report_text(f"Topic: {topic}\n\n{neutral_summary}")


def close_task(task: Task):
    """
    Closes the ClearML task after the pipeline finishes.

    Args:
        task (Task): Active ClearML task
    """

    task.close()
package com.izum1k.pocopaw.ui.component

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyListScope
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.nestedscroll.nestedScroll
import androidx.compose.ui.unit.dp

/**
 * The frame of every pushed page: large flexible bar that collapses with the list, a centred
 * reading column on wide windows, and room under the last row for the navigation bar.
 */
@Composable
fun PawPage(
    title: String,
    modifier: Modifier = Modifier,
    content: LazyListScope.() -> Unit,
) {
    val scroll = TopAppBarDefaults.exitUntilCollapsedScrollBehavior()
    Scaffold(
        modifier = modifier.nestedScroll(scroll.nestedScrollConnection),
        topBar = { PawLargeTopBar(title, scroll) },
        containerColor = MaterialTheme.colorScheme.surface,
    ) { padding ->
        Box(Modifier.fillMaxSize().padding(top = padding.calculateTopPadding()), contentAlignment = Alignment.TopCenter) {
            LazyColumn(
                modifier = Modifier.readingWidth(),
                contentPadding = PaddingValues(start = 16.dp, end = 16.dp, top = 8.dp, bottom = padding.calculateBottomPadding() + 32.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
                content = content,
            )
        }
    }
}
